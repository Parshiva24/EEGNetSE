# Changes from the original notebooks

This refactor consolidates the fifteen Jupyter notebooks and two `.py` utility
files (now archived under [`legacy/`](legacy/)) into the installable `eegnetse`
package. Behaviour was preserved except for the deliberate improvements below.
Because the paper revision re-runs every experiment from scratch, this is the
right moment to correct these.

## Behaviour-affecting changes (re-run experiments to refresh numbers)

1. **Logits instead of double soft-max.** The notebooks returned `F.softmax(...)`
   from `forward` *and* trained with `nn.CrossEntropyLoss` (which internally
   applies log-soft-max), so soft-max was effectively applied twice. The models
   now return raw **logits**, correctly paired with `CrossEntropyLoss`.
   *Predictions are `argmax`-based, so this does not change any reported
   accuracy at evaluation time — only the training gradients (for the better).*

2. **Notch filter applied consistently.** The EEGNet path notch-filtered at
   50 Hz; the FBCNet path did not. Preprocessing is now unified in
   `eegnetse.data.preprocessing`, with the notch on by default for every model
   (`notch_freq=None` disables it).

3. **Stratified validation split.** The 90/10 train/validation split used for
   early stopping is now class-stratified (`stratify=Y`). The notebooks used an
   unstratified random split. (Confirms checklist item B.)

4. **Single, consistent early-stopping rule.** The two divergent policies
   (`min_delta=1e-11` on val-loss in some notebooks, `1e-3` on val-accuracy with
   an `epoch>100 & acc>70` gate in others) are unified in
   `eegnetse.training.EarlyStopping`, which monitors **validation accuracy** with
   configurable `patience`, `warmup_epochs` and `min_acc`. Defaults reproduce the
   base-pretraining gate; fine-tuning uses `warmup_epochs=0`.

## Non-behavioural changes (identical numerics)

- **Vectorised electrode-ranking SE.** `ElectrodeRankingSE` replaces the original
  per-feature-map Python loop with one batched matmul. The arithmetic is
  identical (the two FC weights were already shared across maps); it is just
  much faster.
- **One model definition** for EEGNet and EEGNet-SE, selected by the
  `use_electrode_se` / `use_filter_se` flags, so the "proposed without ranking
  layers" baseline is the same code path.
- Checkpoints are kept in memory during training and restored at the end, rather
  than round-tripping through hard-coded `.pth` filenames.

## Not yet implemented (experiment phase)

- **MVCA baseline** (`eegnetse/models/mvca.py`) — stub raising `NotImplementedError`.
- Revision experiment scripts A1–A6 (train-from-scratch, BN ablation, LOSO-CV,
  complexity, EEG-example figure) — the protocol supports them; the thin runners
  are still to be written.
