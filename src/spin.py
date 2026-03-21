"""
KAIL — SPIN Trainer (Self-Play Fine-Tuning)
Stage 1 of the H-AZR pipeline.

Paper: "Self-Play Fine-Tuning Converts Weak Language Models to Strong"
       Chen et al., 2024 — https://arxiv.org/abs/2401.01335

Core idea:
    At iteration t, the model M_t is trained to distinguish its own responses
    from those of M_{t-1} (the previous version of itself), using a DPO-style
    objective. No human preference labels needed — the model bootstraps quality
    from its own improving outputs.

    Loss = -log σ( β * (log M_t(y_t|x) - log M_{t-1}(y_t|x))
                  - β * (log M_t(y_{t-1}|x) - log M_{t-1}(y_{t-1}|x)) )

    where:
        y_t     = response sampled from M_t (chosen)
        y_{t-1} = response sampled from M_{t-1} (rejected)
        β       = DPO temperature

Usage:
    from src.spin import SPINTrainer

    trainer = SPINTrainer(model, tokenizer, config)
    trainer.run(prompts, n_iterations=3)
    trainer.save("checkpoints/spin_final")
"""

import os
import json
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizer
from datasets import Dataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SPINConfig:
    # Training
    n_iterations: int = 3
    n_samples: int = 256
    beta: float = 0.1              # DPO temperature
    learning_rate: float = 5e-5
    num_train_epochs: int = 1
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.1
    lr_scheduler: str = "cosine"
    max_length: int = 512
    max_prompt_length: int = 256
    fp16: bool = True

    # Generation (for building the dataset each iteration)
    gen_max_new_tokens: int = 400
    gen_temperature: float = 0.8
    gen_top_p: float = 0.95

    # Output
    output_dir: str = "checkpoints/spin"
    save_each_iteration: bool = True
    logging_steps: int = 10

    # Checkpointing (for Colab — saves to Drive)
    checkpoint_base: str = "/content/drive/MyDrive/KAIL/checkpoints"


# ---------------------------------------------------------------------------
# SPIN Trainer
# ---------------------------------------------------------------------------

class SPINTrainer:
    """
    Self-Play Fine-Tuning trainer.

    Args:
        model: PEFT-wrapped model (already prepared for k-bit training)
        tokenizer: matching tokenizer
        config: SPINConfig
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: Optional[SPINConfig] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or SPINConfig()
        self._iteration_responses: List[str] = []   # cache of last iter responses
        self.metrics_history: List[Dict] = []

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate_responses(
        self,
        prompts: List[str],
        temperature: Optional[float] = None,
        show_progress: bool = True,
    ) -> List[str]:
        """Sample one response per prompt from the current model."""
        cfg = self.config
        temp = temperature if temperature is not None else cfg.gen_temperature
        responses = []
        iterator = tqdm(prompts, desc="Generating", disable=not show_progress)

        for prompt in iterator:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=cfg.max_prompt_length,
            ).to(self.model.device)

            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=cfg.gen_max_new_tokens,
                    do_sample=(temp > 0),
                    temperature=max(temp, 1e-4),
                    top_p=cfg.gen_top_p,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            response = self.tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            responses.append(response)

        return responses

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    def build_dataset(
        self,
        prompts: List[str],
        prev_responses: Optional[List[str]] = None,
    ) -> Tuple[Dataset, List[str]]:
        """
        Build one SPIN iteration dataset.

        Returns:
            dataset: HuggingFace Dataset with 'prompt', 'chosen', 'rejected'
            current_responses: responses generated this iteration (→ rejected next iter)
        """
        cfg = self.config

        # Expand prompts to n_samples
        n = cfg.n_samples
        expanded = (prompts * (n // len(prompts) + 1))[:n]

        # Chosen: current model responses
        current_responses = self.generate_responses(expanded)

        # Rejected: previous model responses (or a second sample on iter 0)
        if prev_responses is None:
            # First iteration: generate a second sample as baseline "rejected"
            rejected = self.generate_responses(expanded, temperature=0.3)
        else:
            # Subsequent iterations: use cached responses from last iter
            prev_expanded = (prev_responses * (n // len(prev_responses) + 1))[:n]
            rejected = prev_expanded

        dataset = Dataset.from_dict({
            "prompt": expanded,
            "chosen": current_responses,
            "rejected": rejected,
        })

        return dataset, current_responses

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _train_one_iteration(
        self,
        dataset: Dataset,
        iteration: int,
    ) -> Dict:
        """Run one DPO training pass over the SPIN dataset."""
        from trl import DPOConfig as TRLDPOConfig, DPOTrainer

        cfg = self.config
        iter_dir = os.path.join(
            cfg.checkpoint_base,
            f"spin_iter_{iteration}",
        ) if cfg.save_each_iteration else os.path.join(cfg.output_dir, f"iter_{iteration}")

        os.makedirs(iter_dir, exist_ok=True)

        dpo_config = TRLDPOConfig(
            output_dir=iter_dir,
            num_train_epochs=cfg.num_train_epochs,
            per_device_train_batch_size=cfg.per_device_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            warmup_ratio=cfg.warmup_ratio,
            lr_scheduler_type=cfg.lr_scheduler,
            fp16=cfg.fp16,
            beta=cfg.beta,
            max_length=cfg.max_length,
            max_prompt_length=cfg.max_prompt_length,
            logging_steps=cfg.logging_steps,
            save_steps=100,
            save_total_limit=1,
            report_to="none",
            remove_unused_columns=False,
        )

        trainer = DPOTrainer(
            model=self.model,
            args=dpo_config,
            train_dataset=dataset,
            processing_class=self.tokenizer,
        )

        train_result = trainer.train()
        metrics = {
            "iteration": iteration,
            "train_loss": round(train_result.training_loss, 4),
            "train_steps": train_result.global_step,
        }

        if cfg.save_each_iteration:
            self.model.save_pretrained(iter_dir)
            self.tokenizer.save_pretrained(iter_dir)
            print(f"  Checkpoint saved: {iter_dir}")

        return metrics

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        prompts: List[str],
        n_iterations: Optional[int] = None,
    ) -> List[Dict]:
        """
        Run the full SPIN pipeline.

        Args:
            prompts: seed prompts for self-play
            n_iterations: number of iterations (overrides config if provided)

        Returns:
            metrics_history: list of per-iteration metrics dicts
        """
        n_iter = n_iterations or self.config.n_iterations

        print(f"Starting SPIN: {n_iter} iterations, {self.config.n_samples} samples/iter")
        print(f"Model: {getattr(self.model.config, '_name_or_path', 'unknown')}")
        print(f"β = {self.config.beta} | lr = {self.config.learning_rate}")

        prev_responses = None

        for i in range(1, n_iter + 1):
            print(f"\n{'='*55}")
            print(f"SPIN Iteration {i}/{n_iter}")
            print(f"{'='*55}")

            # 1. Build dataset for this iteration
            print(f"Building dataset ({self.config.n_samples} pairs)...")
            dataset, current_responses = self.build_dataset(prompts, prev_responses)

            # 2. Cache current responses → they become "rejected" next iter
            prev_responses = current_responses

            # 3. Train
            print(f"Training...")
            metrics = self._train_one_iteration(dataset, iteration=i)
            self.metrics_history.append(metrics)
            print(f"  Loss: {metrics['train_loss']:.4f} | Steps: {metrics['train_steps']}")

        return self.metrics_history

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str):
        """Save final SPIN model and training history."""
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

        history_path = os.path.join(path, "spin_metrics.json")
        with open(history_path, "w") as f:
            json.dump(self.metrics_history, f, indent=2)

        print(f"\nSPIN model saved to: {path}")
        print(f"Training history: {history_path}")

    def print_summary(self):
        """Print per-iteration loss summary."""
        print(f"\n{'='*40}")
        print("SPIN Training Summary")
        print(f"{'='*40}")
        for m in self.metrics_history:
            print(f"  Iteration {m['iteration']:2d} | loss: {m['train_loss']:.4f}")
        if self.metrics_history:
            losses = [m["train_loss"] for m in self.metrics_history]
            delta = losses[0] - losses[-1]
            print(f"  Loss improvement: {delta:+.4f} ({losses[0]:.4f} → {losses[-1]:.4f})")
        print(f"{'='*40}")


# ---------------------------------------------------------------------------
# Convenience: default seed prompts for code training
# ---------------------------------------------------------------------------

DEFAULT_CODE_PROMPTS = [
    "Explain what a Python decorator is and write a simple example.",
    "Write a function that implements quicksort in Python.",
    "What is the difference between a list and a tuple in Python?",
    "Write a class that implements a binary search tree in Python.",
    "Explain Big O notation with concrete examples.",
    "Write a Python function that parses a JSON file and returns a dict.",
    "What is recursion? Show a recursive implementation of merge sort.",
    "Write a Python generator that yields Fibonacci numbers indefinitely.",
    "Explain the difference between shallow copy and deep copy in Python.",
    "Write a Python function that validates an email address using regex.",
    "What are Python context managers? Write a custom one with __enter__ and __exit__.",
    "Implement a LRU cache in Python without using functools.",
    "Write a function that finds all permutations of a string.",
    "Explain Python's GIL and when it matters for concurrency.",
    "Write a function that rotates a matrix 90 degrees clockwise in-place.",
    "Implement a simple token bucket rate limiter in Python.",
    "Write a Python function that compresses a string using run-length encoding.",
    "Implement a min-heap class in Python from scratch.",
    "Write a function that detects a cycle in a linked list.",
    "Explain the difference between threading and multiprocessing in Python.",
]
