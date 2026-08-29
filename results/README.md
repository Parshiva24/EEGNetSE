# Results

**Detailed per-subject write-up:** [`model_performance.md`](model_performance.md)
(base / fine-tuned / scratch, with method notes and manuscript cross-reference).

## Table III — Ablation: from-scratch vs. transfer learning

Reviewer response (Section 3.2 of the revision). Three training regimes evaluated
on each online subject's held-out session (S08–S20, 48 trials each). Reproduce with:

```bash
python -m eegnetse.train --stage ablation --subjects 8-20
```

| Regime | Description | Mean ± SD (n=13) |
|--------|-------------|------------------|
| (a) from-scratch | Full EEGNet-SE trained on the subject's 72 calibration trials only (random init) | **54.01 ± 7.77 %** |
| (b) base-only | Subject-independent base (pre-trained on S01–S07), no fine-tuning | **57.85 ± 7.72 %** |
| (c) two-stage (proposed) | Base + fine-tuning of SE (+dense) layers on the subject's calibration data | **57.53 ± 7.82 %** |

Per-subject numbers: [`table3_ablation.csv`](table3_ablation.csv).

### Reading of the results

* **Transfer learning clearly beats from-scratch.** Regime (a) is the lowest by
  ~3.5–3.8 pts, consistent with overfitting when a full EEGNet-SE is trained on
  only ~29 effective trials/class. This is the main point the reviewer asked about.
* **Regimes (b) and (c) are comparable** (57.85 vs 57.53 %; the difference is
  within noise, and (b) is nominally higher). Regimes (b) and (c) are computed
  from the fixed committed checkpoints, so this is not seed variance. The
  fine-tuning stage does **not** raise mean accuracy over the well-generalised
  base; its role is subject-specific SE adaptation (electrode / feature-map
  importance), not a raw accuracy gain.

> Note for the manuscript: the drafted response stated regime (c) "achieves the
> best performance". These numbers do not support (c) > (b); they support
> (a) ≪ (b) ≈ (c). Recommend softening the Table III discussion accordingly.

### Method notes

* (a) from-scratch: same EEGNet-SE architecture (8432 params), random init, all
  layers trainable, Adam lr 1e-3, early stopping on a stratified 20 % validation
  split, evaluated on the online session. Single seed (0); from-scratch is the
  highest-variance regime, so multi-seed averaging (`--seed`) would firm up (a).
* (b), (c): committed checkpoints in `../trained_model/` (deterministic).
* Preprocessing identical across regimes: baseline correction (2500→2000) + notch
  50 Hz + band-pass [0.5, 45] Hz.
