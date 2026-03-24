# KAIL — How to run notebooks on Google Colab

Every notebook in this repo can be launched directly from GitHub
without downloading anything. Three methods below, fastest first.

---

## Method 1 — One-click badge (recommended)

Each notebook has an "Open in Colab" badge in its header.
Click it → Colab opens the notebook directly from GitHub.

The URL format is:
```
https://colab.research.google.com/github/kora-ai-lab/h-azr/blob/main/notebooks/00_setup.ipynb
```

Badges for all notebooks:

| Notebook | Launch |
|----------|--------|
| 00 — Setup | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kora-ai-lab/h-azr/blob/main/notebooks/00_setup.ipynb) |
| 01 — Baseline eval | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kora-ai-lab/h-azr/blob/main/notebooks/01_baseline_eval.ipynb) |
| 02 — H-Neuron probe | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kora-ai-lab/h-azr/blob/main/notebooks/02_h_neuron_probe.ipynb) |
| 03 — SPIN warmup | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kora-ai-lab/h-azr/blob/main/notebooks/03_spin_warmup.ipynb) |
| 04 — H-AZR training | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kora-ai-lab/h-azr/blob/main/notebooks/04_h_azr_training.ipynb) |
05 - master: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kora-ai-lab/h-azr/blob/main/notebooks/KAIL_MASTER.ipynb)

---

## Method 2 — From colab.research.google.com

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **File → Open notebook**
3. Select the **GitHub** tab
4. Paste: `kora-ai-lab/h-azr`
5. Pick the notebook you want → Open

---

## Method 3 — From github.com

1. Navigate to any `.ipynb` file in the `notebooks/` folder
2. At the top of the file preview, click **Open in Colab**
   (GitHub shows this button automatically for notebooks)

---

## Before you run: checklist

```
[ ] Runtime set to T4 GPU
    → Runtime > Change runtime type > T4 GPU

[ ] Run 00_setup.ipynb first (installs deps, mounts Drive)

[ ] HuggingFace token ready
    → huggingface.co/settings/tokens > New token (read access is enough)

[ ] Google Drive mounted (notebook 00 does this automatically)
    → Checkpoints will be saved to: My Drive/KAIL/checkpoints/
```

---

## Handling Colab session resets

Free Colab disconnects after ~12h. Your model weights are lost unless
you saved to Google Drive. Notebook 00 mounts Drive automatically.

Every training notebook saves checkpoints to:
```
/content/drive/MyDrive/KAIL/checkpoints/
```

To resume after a reset:
1. Re-run notebook 00 (reinstalls deps + remounts Drive)
2. In the training notebook, point `checkpoint` to your saved path:
   ```python
   # Instead of loading from HuggingFace:
   MODEL_NAME = "/content/drive/MyDrive/KAIL/checkpoints/spin_iter_2"
   ```

---

## Kaggle alternative (30h/week free, 2×T4)

Kaggle gives 30h of GPU per week for free — more than Colab.
For the longer training runs (notebooks 03 and 04), consider Kaggle:

1. Go to [kaggle.com/code](https://kaggle.com/code) → New Notebook
2. Settings → Accelerator → GPU T4 x2
3. In the first cell:
   ```python
   # Clone the repo
   !git clone https://github.com/kora-ai-lab/h-azr.git
   %cd h-azr
   !pip install -r requirements.txt
   ```
4. Then run any notebook cell-by-cell by copying the cells in

---

## RunPod ($16 budget — use strategically)

Use RunPod only for the final training run of Scenario C (longest run).

```
Recommended pod: RTX 4090 (24GB) — ~$0.44/h
Estimated cost for full Scenario C: ~$6-8
Remaining budget after: ~$8-10 for Scenario D + buffer
```

Setup on RunPod:
```bash
git clone https://github.com/kora-ai-lab/h-azr.git
cd h-azr
pip install -r requirements.txt
# Run training
python -c "
from src.spin import SPINTrainer, DEFAULT_CODE_PROMPTS, SPINConfig
# ... (see notebooks/03 for full setup)
"
```

---

*KAIL — Kora AI Lab | github.com/kora-ai-lab*
