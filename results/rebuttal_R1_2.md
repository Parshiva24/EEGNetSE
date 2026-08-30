# Rebuttal — R1.2 (two-stage transfer vs. from-scratch)

**Reviewer comment R1.2.** The calibration session provides 72 trials per subject
(36 per class), used for fine-tuning the pre-trained subject-independent model,
while the online test session provides only 48 trials. Given that the calibration
data alone contain a sufficient number of trials for training a subject-specific
model from scratch, could the authors clarify why the two-stage approach
(pre-training on S01–S07 followed by fine-tuning on target subjects) is preferred
over directly training a subject-specific model using only the calibration data
from each target subject? Additionally, what is the performance difference between
these two strategies?

---

## Response

We thank the reviewer for this valuable question, which prompted us to add a direct
empirical comparison against a subject-specific model trained from scratch. The
two-stage design is motivated by two considerations that are well established for
EEG deep learning, and which the new ablation confirms.

**1. Data scarcity per subject.** The 72 calibration trials (36 per class) sit at
the lower bound for training an EEGNet-style network from scratch. After the
stratified train/validation split, roughly 29 trials per class remain to fit a
model with ~8.4k parameters, which is prone to overfitting. Cross-subject
pre-training instead supplies an initialisation whose temporal and spatial
convolution filters already encode motor-imagery-relevant structure, so the small
per-subject dataset is used only to *adapt* the model rather than to *learn it from
zero*.

**2. Regularisation through shared features.** Pre-training on S01–S07 constrains
the convolution filters to features that generalise across subjects. Fine-tuning
then updates only the squeeze-and-excitation (SE) electrode/feature-map ranking
layers — a small parameter subset — which is stable to estimate from 72 trials and
keeps the convolutional backbone fixed.

### Ablation

To quantify the performance difference the reviewer asks about, we trained a
subject-specific model **from scratch** for each target subject (S08–S20), using
only that subject's 72 calibration trials, with the identical EEGNet-SE
architecture and preprocessing (5th-order Butterworth band-pass [0.5, 90] Hz,
50 Hz notch) as the proposed model (random initialisation, all layers trainable).
It is evaluated on the same held-out online session (48 trials, 24 per class). The
result is contrasted with the proposed two-stage model as reported in Table II:

| Training strategy | Mean ± SD accuracy (n = 13) |
|-------------------|:---------------------------:|
| Subject-specific, from scratch (calibration only) | 54.17 ± 8.76 % |
| **Proposed two-stage (pre-train + SE fine-tuning)** | **58.65 ± 8.23 %** (Table II) |

Training from scratch is **~4.5 percentage points lower** than the proposed
two-stage model. This is consistent with overfitting when a full EEGNet-SE is fit
to only ~29 effective trials per class, and directly answers the reviewer's
question about the performance difference between the two strategies: the
cross-subject pre-training in the two-stage approach provides a substantially
stronger, better-regularised starting point than subject-specific training from
scratch can achieve at this trial count. The proposed model then fine-tunes only
the SE ranking layers, which additionally yields the per-subject electrode and
feature-map importance maps used in the interpretability analysis (Section 3.2) —
information that a model trained from scratch does not expose.

### Change to the manuscript

Added a paragraph to Section 3.1 reporting the from-scratch ablation (54.17 ±
8.76 %) alongside the proposed two-stage result (58.65 ± 8.23 %, Table II), and a
short discussion attributing the ~4.5-point gap to overfitting of the full model on
limited per-subject calibration data.
