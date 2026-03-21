"""
KAIL — H-AZR Reward Function
Combines AZR code-execution accuracy with H-Neuron hallucination penalty.

R_total = R_accuracy - lambda_h * H_neuron_activation_penalty
"""

import torch
import subprocess
import tempfile
import os
import textwrap
from typing import List, Optional, Dict


def execute_code(code: str, timeout: int = 5) -> bool:
    """
    Execute Python code and return True if it runs without error.
    Used as the binary correctness signal in AZR.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(textwrap.dedent(code))
        tmp_path = f.name

    try:
        result = subprocess.run(
            ["python", tmp_path],
            capture_output=True,
            timeout=timeout,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        os.unlink(tmp_path)


def accuracy_reward(completions: List[str], solutions: List[str]) -> List[float]:
    """
    Binary reward: 1.0 if code executes correctly, 0.0 otherwise.
    This is the core AZR reward signal.
    """
    rewards = []
    for completion, solution in zip(completions, solutions):
        # Combine the generated code with a test assertion
        test_code = f"{completion}\n\n# Verification\n{solution}"
        rewards.append(1.0 if execute_code(test_code) else 0.0)
    return rewards


def format_reward(completions: List[str]) -> List[float]:
    """
    Soft reward for following the expected code format.
    Encourages structured outputs.
    """
    rewards = []
    for completion in completions:
        score = 0.0
        if "```python" in completion or "def " in completion:
            score += 0.3
        if "return" in completion:
            score += 0.2
        if completion.strip().endswith("```") or completion.strip().endswith("\n"):
            score += 0.1
        rewards.append(min(score, 0.5))  # cap at 0.5
    return rewards


class HAZRReward:
    """
    H-AZR composite reward function.

    R_total = R_accuracy(code_execution)
            + R_format(output_structure)
            - lambda_h * H_neuron_penalty(activations)

    Args:
        lambda_h: weight for hallucination penalty (default 0.1)
        h_neurons: H-Neuron dict from HNeuronProbe.identify()
        use_format_reward: whether to include format soft reward
    """

    def __init__(
        self,
        lambda_h: float = 0.1,
        h_neurons: Optional[Dict] = None,
        use_format_reward: bool = True,
    ):
        self.lambda_h = lambda_h
        self.h_neurons = h_neurons
        self.use_format_reward = use_format_reward

    def __call__(
        self,
        completions: List[str],
        solutions: List[str],
        hidden_states: Optional[Dict[int, torch.Tensor]] = None,
    ) -> List[float]:
        """
        Compute H-AZR reward for a batch of completions.

        Args:
            completions: list of model-generated code strings
            solutions: list of verification/test code strings
            hidden_states: {layer_idx: tensor} from forward pass (for H-Neuron penalty)

        Returns:
            List of reward scalars, one per completion
        """
        n = len(completions)

        # 1. Accuracy reward (binary, from code execution)
        acc_rewards = accuracy_reward(completions, solutions)

        # 2. Format reward (optional soft signal)
        if self.use_format_reward:
            fmt_rewards = format_reward(completions)
        else:
            fmt_rewards = [0.0] * n

        # 3. H-Neuron penalty
        h_penalties = [0.0] * n
        if self.h_neurons is not None and hidden_states is not None:
            from src.h_neurons import HNeuronProbe
            # Compute penalty per sample in batch
            for i in range(n):
                sample_states = {
                    k: v[i:i+1] for k, v in hidden_states.items()
                }
                dummy_probe = HNeuronProbe.__new__(HNeuronProbe)
                dummy_probe.h_neurons = self.h_neurons
                penalty = dummy_probe.compute_penalty(sample_states)
                h_penalties[i] = penalty.item()

        # 4. Combine
        rewards = []
        for i in range(n):
            r = acc_rewards[i] + fmt_rewards[i] - self.lambda_h * h_penalties[i]
            rewards.append(r)

        return rewards

    def log_stats(self, rewards: List[float], acc: List[float], penalties: List[float]):
        """Return dict of reward component stats for W&B logging."""
        return {
            "reward/total_mean": sum(rewards) / len(rewards),
            "reward/accuracy_mean": sum(acc) / len(acc),
            "reward/h_penalty_mean": sum(penalties) / len(penalties),
            "reward/accuracy_rate": sum(1 for r in acc if r > 0) / len(acc),
        }
