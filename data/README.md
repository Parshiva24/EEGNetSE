# MI Direction-Decoding Dataset (`mitrials`)

Per-subject motor-imagery (MI) **left/right direction** EEG trials used to train and
evaluate the EEGNet-SE model (manuscript sensors-4477952). This is the ready-to-use
form consumed directly by the [`eegnetse`](../implmentations/EEGNetSE) package — no
conversion needed.

## Files

`S01_mitrials.mat` … `S20_mitrials.mat` (20 subjects). Each holds:

| Variable | Shape | Meaning |
|---|---|---|
| `Xtrain` | `(72, 2500, 27)` | calibration trials — 72 trials × 2500 samples × 27 channels |
| `Ytrain` | `(72, 1)` | true direction label: **0 = left, 1 = right** |
| `Ytrain_fb` | `(72, 1)` | label shown by the closed-loop feedback |
| `Xtest` | `(48, 2500, 27)` or `NaN` | online-session trials (`NaN` if no online session) |
| `Ytest` | `(48, 1)` or `NaN` | online true labels |
| `Ytest_fb` | `(48, 1)` or `NaN` | online feedback labels |

- **Sampling rate:** 500 Hz. **2500 samples** = 500-sample (1 s) pre-cue baseline +
  2000-sample (4 s) MI window. Baseline correction subtracts and removes the first
  500 samples, leaving the 2000-sample model input.
- **27 channels** (BrainProducts actiCHamp, order):
  FP1, FP2, AF1, AF3, AFz, AF4, AF8, F3, F1, Fz, F2, F4, FC3, FC1, FCz, FC2, FC4,
  C3, C1, Cz, C2, C4, CP3, CP1, CPz, CP2, CP4.
- **Amplitudes** are in microvolts (already scaled).

## Train / evaluation split

- **S01–S07** — calibration session only (`Xtest = NaN`). Pooled = **504 trials**,
  used to pre-train the subject-independent base model.
- **S08–S20** — have an online session; used for subject-specific fine-tuning
  (on `Xtrain`) and evaluation (on `Xtest`, never seen in training).

## Subject key

| S## | Name | Session |
|---|---|---|
| S01 | Asish | calib only |
| S02 | Danish | calib only |
| S03 | Gayathri | calib only |
| S04 | Navaneeth | calib only |
| S05 | Nithyasree | calib only |
| S06 | Rakesh | calib only |
| S07 | Shanmu | calib only |
| S08 | Ajul | calib + online |
| S09 | Ananthu | calib + online |
| S10 | Aswathy | calib + online |
| S11 | Athira | calib + online |
| S12 | Bharath2 | calib + online |
| S13 | Durga | calib + online |
| S14 | Greeshma | calib + online |
| S15 | Jijomon | calib + online |
| S16 | Kumudini | calib + online |
| S17 | Mithul | calib + online |
| S18 | Pramod1 | calib + online |
| S19 | Sagila | calib + online |
| S20 | Venkatesh | calib + online |

## Usage

Point the package at this folder:

```bash
export EEGNETSE_DATA="E:/PostDoc@SIT/SIT2024/MDPI2026/dataset"
```

then run the training scripts (see the package README). Provenance: exported from the
EEGLAB epoched recordings (`epoch_extraction_usingEEGLAB.m`); 27 channels selected on
the raw montage; identical to `D:\PostDoc@SIT\SIT2024\MIDecoding_SENet\data`.
