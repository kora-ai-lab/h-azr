<p align="center">
  <img src="https://raw.githubusercontent.com/kora-ai-lab/.github/main/profile/kail-logo-banner.svg" alt="KAIL" width="500"/>
</p>

<h1 align="center">H-AZR: Hallucination-Aware Self-Play Reasoning</h1>

<p align="center">
  <em>A 3-stage training pipeline combining SPIN warm-up, H-Neuron probing, and GRPO self-play with hallucination penalty</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/status-research%20in%20progress-EF9F27?style=flat-square"/>
  <img src="https://img.shields.io/badge/compute-colab%20T4%20(free)-1D9E75?style=flat-square"/>
  <img src="https://img.shields.io/badge/base%20model-Qwen2.5--Coder--1.5B-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/lab-KAIL%20%7C%20Kora%20AI%20Lab-0D0D0D?style=flat-square"/>
</p>

---

## Research Hypothesis

> Standard self-play reasoning (AZR) improves accuracy but may reinforce hallucination by rewarding confident-sounding outputs. We hypothesize that penalizing H-Neuron activation during GRPO training produces a model that reasons better **and** knows when it doesn't know.

**Core reward function:**
```
R_total = R_accuracy(code_execution) − λ × mean(H_neuron_activations)
```

---

## The 4 Experimental Scenarios

| Scenario | Pipeline | Purpose |
|----------|----------|---------|
| **A** — Baseline | Base model only | Measure raw hallucination + reasoning |
| **B** — SPIN only | Base → SPIN | Does alignment reduce H-Neurons? |
| **C** — Full H-AZR | Base → SPIN → H-AZR | Main contribution |
| **D** — H-AZR no warmup | Base → H-AZR | Is SPIN warm-up necessary? |

All 4 scenarios run on **free Colab T4 (16GB VRAM)** with Qwen2.5-Coder-1.5B + QLoRA 4-bit.

---

## Papers This Builds On

| Paper | Key contribution we use |
|-------|------------------------|
| [AZR (2025)](https://arxiv.org/abs/2505.03335) | GRPO self-play, zero external data |
| [SPIN (2024)](https://arxiv.org/abs/2401.01335) | Iterative self-distillation warm-up |
| [BitNet b1.58 (2023)](https://arxiv.org/abs/2310.11453) | Ternary model target (Phase 2) |
| H-Neurons (2024) | <0.1% neurons predict hallucination onset |

---

## Repository Structure

```
h-azr/
├── notebooks/
│   ├── 01_baseline_eval.ipynb        # Colab: load model, eval hallucination
│   ├── 02_h_neuron_probe.ipynb       # Colab: identify H-Neurons
│   ├── 03_spin_warmup.ipynb          # Colab: Stage 1 SPIN training
│   └── 04_h_azr_training.ipynb       # Colab: Stage 3 H-AZR + reward
├── src/
│   ├── h_neurons.py                  # H-Neuron probe + activation tracking
│   ├── spin.py                       # SPIN trainer
│   ├── reward.py                     # H-AZR reward function
│   └── eval.py                       # Evaluation metrics
├── configs/
│   └── training.yaml                 # All hyperparameters
├── paper/
│   └── h_azr_draft.tex               # LaTeX paper draft
├── assets/
│   └── figures/                      # Paper figures
└── requirements.txt
```

---

## Quickstart (Colab)

```python
# Open notebook 01 first:
# https://colab.research.google.com/github/kora-ai-lab/h-azr/blob/main/notebooks/01_baseline_eval.ipynb
```

All notebooks are self-contained — just click "Open in Colab" and run.

---

## Results (in progress)

| Scenario | TruthfulQA | HalluEval | GSM8K | Status |
|----------|-----------|-----------|-------|--------|
| A — Baseline | — | — | — | 🔄 Running |
| B — SPIN | — | — | — | ⏳ Pending |
| C — Full H-AZR | — | — | — | ⏳ Pending |
| D — H-AZR no warmup | — | — | — | ⏳ Pending |

---

## Citation

```bibtex
@misc{kail2025hazr,
  title   = {H-AZR: Hallucination-Aware Self-Play Reasoning via H-Neuron Penalized GRPO},
  author  = {Kora AI Lab},
  year    = {2025},
  url     = {https://github.com/kora-ai-lab/h-azr}
}
```

---

<p align="center">
  Built in West Africa 🌍 · <a href="https://github.com/kora-ai-lab">Kora AI Lab</a>
</p>
