"""
KAIL — H-Neuron Probe
Identifies neurons responsible for hallucination onset.

Based on: "Distinguishing Ignorance from Error in LLM Hallucinations" (2024)
Key finding: <0.1% of neurons predict hallucination with high precision.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from transformers import PreTrainedModel, PreTrainedTokenizer
from datasets import load_dataset
from tqdm import tqdm
import json


class HNeuronProbe:
    """
    Identifies H-Neurons (hallucination-predictive neurons) in a language model.

    Usage:
        probe = HNeuronProbe(model, tokenizer)
        h_neurons = probe.identify(dataset, threshold=2.0)
        penalty = probe.compute_penalty(hidden_states, h_neurons)
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.h_neurons: Optional[Dict] = None
        self._hooks = []

    def _get_mlp_activations(self, input_ids: torch.Tensor) -> Dict[int, torch.Tensor]:
        """Forward pass collecting MLP activations at every layer."""
        activations = {}

        def make_hook(layer_idx):
            def hook(module, input, output):
                # output shape: (batch, seq_len, hidden_dim)
                activations[layer_idx] = output.detach().cpu()
            return hook

        hooks = []
        for i, layer in enumerate(self.model.model.layers):
            h = layer.mlp.register_forward_hook(make_hook(i))
            hooks.append(h)

        with torch.no_grad():
            self.model(input_ids.to(self.device))

        for h in hooks:
            h.remove()

        return activations

    def _is_hallucination(self, model_answer: str, ground_truth: str) -> bool:
        """Simple exact-match hallucination check. Override for fuzzy matching."""
        gt = ground_truth.strip().lower()
        ans = model_answer.strip().lower()
        return gt not in ans

    def collect_activation_stats(
        self,
        questions: List[str],
        answers: List[str],
        max_samples: int = 500,
    ) -> Tuple[Dict, Dict]:
        """
        Collect mean MLP activations separately for hallucinated vs correct responses.

        Returns:
            hall_stats: {layer_idx: mean_activation_vector} for hallucinations
            corr_stats: {layer_idx: mean_activation_vector} for correct responses
        """
        print(f"Collecting activations on {min(max_samples, len(questions))} samples...")

        hall_acts: Dict[int, List] = {}
        corr_acts: Dict[int, List] = {}

        for q, gt in tqdm(zip(questions[:max_samples], answers[:max_samples])):
            inputs = self.tokenizer(q, return_tensors="pt", truncation=True, max_length=256)
            input_ids = inputs["input_ids"]

            # Get model prediction
            with torch.no_grad():
                out = self.model.generate(
                    input_ids.to(self.device),
                    max_new_tokens=64,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            pred = self.tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)

            # Get activations on the question (not generation)
            acts = self._get_mlp_activations(input_ids)
            # Take mean over sequence dimension, last token as proxy
            last_acts = {k: v[0, -1, :].numpy() for k, v in acts.items()}

            is_hall = self._is_hallucination(pred, gt)
            target = hall_acts if is_hall else corr_acts

            for layer_idx, act in last_acts.items():
                if layer_idx not in target:
                    target[layer_idx] = []
                target[layer_idx].append(act)

        # Compute means
        hall_means = {k: np.mean(v, axis=0) for k, v in hall_acts.items() if v}
        corr_means = {k: np.mean(v, axis=0) for k, v in corr_acts.items() if v}

        return hall_means, corr_means

    def identify(
        self,
        questions: List[str],
        answers: List[str],
        threshold: float = 2.0,
        max_samples: int = 500,
    ) -> Dict:
        """
        Identify H-Neurons via z-score difference between hallucinated/correct activations.

        Args:
            threshold: z-score threshold. Higher = more selective (default: 2.0)

        Returns:
            Dict mapping layer_idx -> list of neuron indices (H-Neurons)
        """
        hall_means, corr_means = self.collect_activation_stats(questions, answers, max_samples)

        h_neurons = {}
        total_neurons = 0
        total_h = 0

        for layer_idx in hall_means:
            if layer_idx not in corr_means:
                continue

            diff = hall_means[layer_idx] - corr_means[layer_idx]
            std = np.std(diff)
            if std == 0:
                continue

            z_scores = diff / std
            h_idx = np.where(np.abs(z_scores) > threshold)[0].tolist()

            total_neurons += len(diff)
            total_h += len(h_idx)

            if h_idx:
                h_neurons[layer_idx] = {
                    "indices": h_idx,
                    "z_scores": z_scores[h_idx].tolist(),
                    "direction": np.sign(diff[h_idx]).tolist(),
                }

        pct = 100 * total_h / total_neurons if total_neurons > 0 else 0
        print(f"\nH-Neurons identified: {total_h} / {total_neurons} ({pct:.3f}%)")
        print(f"Affected layers: {len(h_neurons)} / {len(hall_means)}")

        self.h_neurons = h_neurons
        return h_neurons

    def compute_penalty(
        self,
        hidden_states: Dict[int, torch.Tensor],
        h_neurons: Optional[Dict] = None,
    ) -> torch.Tensor:
        """
        Compute H-Neuron activation penalty for use in GRPO reward.

        Args:
            hidden_states: {layer_idx: tensor(batch, seq, hidden_dim)}
            h_neurons: H-Neuron dict. Uses self.h_neurons if None.

        Returns:
            penalty: scalar tensor (mean absolute activation of H-Neurons)
        """
        if h_neurons is None:
            h_neurons = self.h_neurons
        if h_neurons is None:
            raise ValueError("No H-Neurons identified. Run .identify() first.")

        penalties = []
        for layer_idx, info in h_neurons.items():
            if layer_idx not in hidden_states:
                continue
            acts = hidden_states[layer_idx]  # (batch, seq, hidden)
            # Mean over batch and seq, select H-Neuron indices
            h_idx = torch.tensor(info["indices"], device=acts.device)
            h_acts = acts[:, :, h_idx]  # (batch, seq, n_h)
            penalties.append(h_acts.abs().mean())

        if not penalties:
            return torch.tensor(0.0)

        return torch.stack(penalties).mean()

    def save(self, path: str):
        """Save H-Neuron map to JSON."""
        if self.h_neurons is None:
            raise ValueError("No H-Neurons to save. Run .identify() first.")
        with open(path, "w") as f:
            json.dump(self.h_neurons, f, indent=2)
        print(f"H-Neurons saved to {path}")

    @classmethod
    def load(cls, path: str) -> Dict:
        """Load H-Neuron map from JSON."""
        with open(path) as f:
            return json.load(f)
