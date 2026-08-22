# Legacy notebooks (archived)

These are the original exploratory notebooks and utility scripts that the
`eegnetse` package was refactored from. They are kept for provenance — the
markdown cells contain previously reported accuracy numbers that are useful
references — but they are **superseded** and are not maintained.

| Legacy file | Replaced by |
|---|---|
| `direction_learning_utils.py` (proposed EEGNet-SE + training) | `eegnetse/models/eegnet.py`, `eegnetse/models/layers.py`, `eegnetse/training/` |
| `direction_utils.py` (I/O + preprocessing) | `eegnetse/data/io.py`, `eegnetse/data/preprocessing.py` |
| `direction_eegnetSE_tl*.ipynb`, `direction_SEPerElectrode*.ipynb` | `scripts/pretrain_base.py`, `scripts/finetune_eval.py` |
| `direction_eegnet*.ipynb` | same scripts with `--model eegnet` |
| `direction_using_fbcnet.ipynb` | `--model fbcnet` |
| `direction_using_senet.ipynb`, `senet_for_electrodes.ipynb`, `direction_using_wavelets_senet.ipynb`, `incremental_training.ipynb`, `gpu_testing.ipynb`, `direction_using_deeplearning.ipynb` | exploratory; not ported |
| `struct2double_midata.m`, `struct_variables_to_double.m` | MATLAB data-prep, unchanged |

See [`../CHANGES.md`](../CHANGES.md) for what changed in the port.
