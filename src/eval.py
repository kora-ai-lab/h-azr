"""
KAIL — Unified Evaluation Script
Runs all benchmarks for all 4 scenarios and produces the paper's Table 1.

Usage:
    # Evaluate a single scenario
    python src/eval.py --checkpoint ./checkpoints/spin_final --scenario B

    # Evaluate all scenarios (requires all checkpoints)
    python src/eval.py --all --base_model Qwen/Qwen2.5-Coder-1.5B-Instruct
"""

import argparse
import json
import os
import subprocess
import tempfile
import torch
import numpy as np
from typing import Dict, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_or_name: str, device: str = "cuda"):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-Coder-1.5B-Instruct", trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_or_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def eval_truthfulqa(model, tokenizer, n_samples: int = 100) -> Dict:
    """TruthfulQA: measures factual accuracy / hallucination rate."""
    dataset = load_dataset("truthful_qa", "generation", split="validation")
    samples = list(dataset.select(range(min(n_samples, len(dataset)))))

    correct = 0
    results = []
    for s in tqdm(samples, desc="TruthfulQA"):
        pred = generate(model, tokenizer, s["question"], max_new_tokens=128)
        hit = s["best_answer"].lower()[:30] in pred.lower()
        correct += int(hit)
        results.append({"q": s["question"], "pred": pred, "correct": hit})

    accuracy = correct / len(samples)
    return {
        "accuracy": round(accuracy, 4),
        "hallucination_rate": round(1.0 - accuracy, 4),
        "n": len(samples),
        "details": results,
    }


def execute_code(code: str, test: str, timeout: int = 5) -> bool:
    """Execute code + test assertions, return True if all pass."""
    full = code + "\n\n# Test\n" + test
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full)
        tmp = f.name
    try:
        r = subprocess.run(["python", tmp], capture_output=True, timeout=timeout)
        return r.returncode == 0
    except Exception:
        return False
    finally:
        os.unlink(tmp)


CODE_TASKS = [
    {"prompt": "Write a Python function `add(a, b)` that returns the sum.",
     "test": "assert add(2, 3) == 5\nassert add(-1, 1) == 0"},
    {"prompt": "Write a Python function `is_palindrome(s)` returning True if s is a palindrome.",
     "test": "assert is_palindrome('racecar')\nassert not is_palindrome('hello')"},
    {"prompt": "Write a Python function `fibonacci(n)` returning the nth Fibonacci (0-indexed).",
     "test": "assert fibonacci(0)==0\nassert fibonacci(1)==1\nassert fibonacci(6)==8"},
    {"prompt": "Write a Python function `count_vowels(s)` returning vowel count.",
     "test": "assert count_vowels('hello')==2\nassert count_vowels('rhythm')==0"},
    {"prompt": "Write a Python function `flatten(lst)` flattening a list of lists.",
     "test": "assert flatten([[1,2],[3,4]])==[1,2,3,4]"},
    {"prompt": "Write a Python function `is_prime(n)` returning True if n is prime.",
     "test": "assert is_prime(7)\nassert not is_prime(4)\nassert is_prime(2)"},
    {"prompt": "Write a Python function `binary_search(lst, target)` returning the index or -1.",
     "test": "assert binary_search([1,3,5,7,9], 5)==2\nassert binary_search([1,3,5], 4)==-1"},
    {"prompt": "Write a Python function `reverse_words(s)` reversing the words in a sentence.",
     "test": "assert reverse_words('hello world')=='world hello'"},
    {"prompt": "Write a Python function `gcd(a, b)` returning the GCD.",
     "test": "assert gcd(12, 8)==4\nassert gcd(7, 3)==1"},
    {"prompt": "Write a Python function `remove_duplicates(lst)` removing duplicates preserving order.",
     "test": "assert remove_duplicates([1,2,2,3,1])==[1,2,3]"},
]


def eval_code(model, tokenizer) -> Dict:
    """Code execution benchmark: pass@1 rate."""
    passed = 0
    results = []
    for task in tqdm(CODE_TASKS, desc="Code eval"):
        code = generate(model, tokenizer, task["prompt"], max_new_tokens=300)
        # Strip markdown fences
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[1].split("```")[0]
        ok = execute_code(code, task["test"])
        passed += int(ok)
        results.append({"task": task["prompt"][:60], "passed": ok})

    pass_rate = passed / len(CODE_TASKS)
    return {
        "pass_rate": round(pass_rate, 4),
        "passed": passed,
        "total": len(CODE_TASKS),
        "details": results,
    }


def eval_h_neurons(model, h_neuron_path: str, n_samples: int = 100) -> Dict:
    """Measure mean H-Neuron activation on a held-out set."""
    if not os.path.exists(h_neuron_path):
        return {"error": f"h_neurons.json not found at {h_neuron_path}"}

    with open(h_neuron_path) as f:
        h_data = json.load(f)
    h_neurons = h_data["layers"]

    dataset = load_dataset("trivia_qa", "rc.nocontext", split="validation")
    samples = list(dataset.select(range(min(n_samples, len(dataset)))))

    penalties = []
    for s in tqdm(samples[:50], desc="H-Neuron activation"):
        activations = {}

        def make_hook(idx):
            def hook(m, inp, out):
                activations[idx] = out.detach().cpu().float()
            return hook

        hooks = []
        try:
            layers = model.base_model.model.model.layers
        except AttributeError:
            layers = model.model.layers

        for i, layer in enumerate(layers):
            hooks.append(layer.mlp.register_forward_hook(make_hook(i)))

        inputs = model.base_model.model.config if hasattr(model, "base_model") else model.config
        tokenizer_tmp = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-Coder-1.5B-Instruct", trust_remote_code=True
        )
        enc = tokenizer_tmp(s["question"], return_tensors="pt", truncation=True, max_length=256)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            model(**enc)
        for h in hooks:
            h.remove()

        layer_pens = []
        for layer_str, info in h_neurons.items():
            li = int(layer_str)
            if li not in activations:
                continue
            acts = activations[li][0, -1, :]
            h_idx = torch.tensor(info["indices"])
            layer_pens.append(acts[h_idx].abs().mean().item())
        if layer_pens:
            penalties.append(float(np.mean(layer_pens)))

    return {
        "mean_h_activation": round(float(np.mean(penalties)), 6) if penalties else None,
        "n": len(penalties),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_eval(
    checkpoint: str,
    scenario: str,
    h_neuron_path: str = "assets/h_neurons.json",
    output_dir: str = "results",
) -> Dict:
    print(f"\n{'='*60}")
    print(f"Evaluating Scenario {scenario}: {checkpoint}")
    print(f"{'='*60}")

    model, tokenizer = load_model(checkpoint)

    metrics = {"scenario": scenario, "checkpoint": checkpoint}

    print("\n[1/3] TruthfulQA...")
    metrics["truthfulqa"] = eval_truthfulqa(model, tokenizer)

    print("\n[2/3] Code eval...")
    metrics["code"] = eval_code(model, tokenizer)

    print("\n[3/3] H-Neuron activation...")
    metrics["h_neurons"] = eval_h_neurons(model, h_neuron_path)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"scenario_{scenario}.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"SCENARIO {scenario} RESULTS")
    print(f"{'='*60}")
    print(f"TruthfulQA accuracy:    {metrics['truthfulqa']['accuracy']:.3f}")
    print(f"Hallucination rate:     {metrics['truthfulqa']['hallucination_rate']:.3f}")
    print(f"Code pass rate:         {metrics['code']['pass_rate']:.3f}")
    if metrics['h_neurons'].get('mean_h_activation') is not None:
        print(f"Mean H-Neuron activ.:   {metrics['h_neurons']['mean_h_activation']:.6f}")
    print(f"\nSaved to: {out_path}")

    return metrics


def print_comparison_table(results_dir: str = "results"):
    """Print Table 1 for the paper from all saved results."""
    scenarios = ["A", "B", "C", "D"]
    labels = {
        "A": "Baseline",
        "B": "SPIN only",
        "C": "Full H-AZR",
        "D": "H-AZR no warmup",
    }

    print(f"\n{'='*70}")
    print("KAIL H-AZR — Table 1: Full Comparison")
    print(f"{'='*70}")
    print(f"{'Scenario':<22} {'TruthfulQA':>12} {'Hall. Rate':>11} {'Code Pass':>10} {'H-Act':>10}")
    print("-" * 70)

    for sc in scenarios:
        path = os.path.join(results_dir, f"scenario_{sc}.json")
        if not os.path.exists(path):
            print(f"{sc} — {labels[sc]:<18} {'(not run)':>12}")
            continue
        with open(path) as f:
            m = json.load(f)
        tqa = m["truthfulqa"]["accuracy"]
        hall = m["truthfulqa"]["hallucination_rate"]
        code = m["code"]["pass_rate"]
        hact = m["h_neurons"].get("mean_h_activation", "—")
        hact_str = f"{hact:.6f}" if isinstance(hact, float) else "—"
        label = f"{sc} — {labels[sc]}"
        print(f"{label:<22} {tqa:>12.3f} {hall:>11.3f} {code:>10.3f} {hact_str:>10}")

    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--scenario", type=str, choices=["A", "B", "C", "D"], help="Scenario label")
    parser.add_argument("--h_neuron_path", default="assets/h_neurons.json")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--table", action="store_true", help="Print comparison table from saved results")
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct",
                        help="Base model name for Scenario A")
    args = parser.parse_args()

    if args.table:
        print_comparison_table(args.output_dir)
    elif args.checkpoint and args.scenario:
        run_eval(args.checkpoint, args.scenario, args.h_neuron_path, args.output_dir)
    elif args.scenario == "A":
        run_eval(args.base_model, "A", args.h_neuron_path, args.output_dir)
    else:
        parser.print_help()
