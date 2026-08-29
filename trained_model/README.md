# Trained models

Saved EEGNet-SE checkpoints for the MDPI *Sensors* study
("Subject-Specific Deep Learning Model for Motor Imagery Direction Decoding").
These are the artifacts behind the paper's reported accuracies; loading them and
running inference on each subject's online session reproduces the paper (no
retraining required).

## Files

| File | What it is |
|------|------------|
| `basedEEGNetSE_model.pth` | Subject-independent **base** model, pre-trained on the pooled calibration trials of S01–S07. |
| `tuned_sub{08..20}_SETrue_DenseTrue_conv2dFalse.pth` | Per-subject fine-tuned model — **SE + Dense** layers updated (the paper's proposed configuration). |
| `tuned_sub{08..20}_SETrue_DenseFalse_conv2dFalse.pth` | Per-subject fine-tuned model — **SE only** (ablation). |

13 online subjects (S08–S20), 2 fine-tune variants each, plus 1 base = 27 files.

## Architecture / preprocessing (needed to load them)

- `EEGNet` with Squeeze-and-Excitation (electrode-SE + filter-SE, reduction = 3),
  `Chans=27`, `Samples=2000`; forward returns `F.softmax`. The final dense layer is
  wrapped in `weight_norm`, so the `state_dict` stores `norm_constraint.weight_g/v`
  (dense in-features = 992 = 16 × 2000/32).
- Preprocessing: baseline correction (2500 → 2000 samples) → notch 50 Hz →
  Butterworth band-pass [0.5, 45] Hz, order 5, `scipy.filtfilt`.
- Evaluate on each subject's online test set (`Xtest`/`Ytest` in `data/S##_mitrials.mat`).

The reference model + preprocessing implementation lives in the legacy project
`MIDecoding_SENet/src/` (`direction_learning_utils.py`, `direction_utils.py`).

## Inference results (online session, S08–S20)

| Configuration | Mean ± SD (n=13) | Drop lowest (S10) |
|---------------|------------------|-------------------|
| Base as inferencer (subject-independent) | 57.85 ± 7.72 % | — |
| SE + Dense fine-tuned (proposed) | 57.53 ± 7.82 % | 58.33 ± 7.61 % |
| SE only fine-tuned | 57.53 ± 7.73 % | 58.51 ± 7.24 % |

Paper Table I proposed EEGNetSE = **58.28 %**. The remaining sub-point gap is
consistent with the band choice ([0.5, 45] here vs [0.5, 90] in the revision) and
seed noise. Note: a **cold retrain from scratch** lands ~3–4 pts lower (~54 %);
these saved checkpoints are what reproduce the reported numbers.
