# EEGNet-SE — Motor-Imagery Direction Decoding

Subject-specific deep learning for **left/right motor-imagery (MI) direction
decoding** from EEG, built on EEGNet with squeeze-and-excitation (SE) layers over
electrodes and feature maps. Reference implementation for the manuscript
*"Subject Specific Deep Learning Model for Motor Imagery Direction Decoding"*
(sensors-4477952).

## The method in one paragraph

A subject-independent **base model** is pre-trained on the pooled calibration
trials of subjects S01–S07. For each target subject (S08–S20) the base model is
then **fine-tuned** on that subject's 72 calibration trials — the SE (and dense)
layers are updated, the rest frozen — and evaluated on that subject's held-out
online (closed-loop feedback) session. No online trial is ever used for training
or model selection.

## Layout

```
EEGNetSE/
├── data/                     # S01_mitrials.mat … S20_mitrials.mat  (git-ignored)
├── trained_model/            # committed checkpoints that reproduce the paper
│   ├── basedEEGNetSE_model.pth
│   └── tuned_sub08..20_SETrue_Dense{True,False}_conv2dFalse.pth
└── eegnetse/                 # the package — three modules
    ├── core.py               # EEGNet-SE model + preprocessing + data loading
    ├── train.py              # base pre-training + per-subject fine-tuning
    └── infer.py              # reproduce paper accuracies from checkpoints
```

## Install

```bash
# install PyTorch for your platform from https://pytorch.org, then:
pip install -e .
```

The dataset is expected in `./data` (files `S01_mitrials.mat` … `S20_mitrials.mat`);
pass `--data-dir` to override.

## 1. Reproduce the paper (inference only)

Load the committed checkpoints and evaluate each subject's online session — no
training required:

```bash
python -m eegnetse.infer
```

This prints per-subject accuracies and means for the base model, the proposed
**SE + Dense** fine-tune, and the **SE-only** ablation:

| Configuration | Mean ± SD (n=13) |
|---|---|
| Base as inferencer (subject-independent) | 57.85 ± 7.72 % |
| SE + Dense fine-tuned (proposed) | 57.53 ± 7.82 % |
| SE only fine-tuned | 57.53 ± 7.73 % |

Paper Table I proposed EEGNet-SE = **58.28 %**. Options: `--config se_dense`
(proposed only), `--model-dir outputs` (evaluate a freshly trained set),
`--band 0.5,90` (override the band-pass).

## 2. Full training

```bash
# both stages: pre-train the base on S01-S07, then fine-tune every subject S08-S20
python -m eegnetse.train --stage all

# individual stages / variants
python -m eegnetse.train --stage base
python -m eegnetse.train --stage finetune --subjects 8-20            # SE + Dense (proposed)
python -m eegnetse.train --stage finetune --no-dense                 # SE only
```

Freshly trained weights go to `outputs/` (never overwriting `trained_model/`
unless you point `--out-dir` there). Evaluate them with
`python -m eegnetse.infer --model-dir outputs`.

## As a library

```python
import eegnetse, torch

# inference
acc = eegnetse.evaluate_checkpoint("trained_model/tuned_sub14_SETrue_DenseTrue_conv2dFalse.pth",
                                   subject=14)

# training
base, _ = eegnetse.pretrain_base()
model, val_acc, n_ft = eegnetse.finetune_subject(base.state_dict(), subject=14, dense=True)
```

## Notes

* Model + preprocessing (`eegnetse/core.py`) are ported from the reference
  implementation that produced the checkpoints, so `load_state_dict` is strict:
  EEGNet-SE (SE reduction 3, soft-max head, weight-normalised dense, 8432 params);
  baseline correction (2500 → 2000 samples) + notch 50 Hz + band-pass [0.5, 45] Hz.
* A **cold retrain from scratch** typically lands ~3–4 pts below the reported
  numbers (~54 %); the committed checkpoints are what reproduce the paper.
* Original exploratory notebooks are archived under [`legacy/`](legacy/).
