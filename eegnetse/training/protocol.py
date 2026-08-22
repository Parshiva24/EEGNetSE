"""Two-stage transfer-learning protocol.

Stage 1 (:func:`pretrain_base`): train a subject-independent base model on the
pooled calibration trials of S01-S07.

Stage 2 (:func:`finetune_subject`): for a target online subject (S08-S20), load
the base model, freeze all but a chosen set of layers, and fine-tune on that
subject's 72 calibration trials. Evaluation (:func:`evaluate_online`) uses only
the held-out online session -- no online trial is ever seen during training or
model selection.
"""

from __future__ import annotations

import copy

from sklearn.model_selection import train_test_split

from eegnetse.data.io import calibration_dataset, online_session
from eegnetse.models import build_model
from eegnetse.training.tensors import make_loader
from eegnetse.training.trainer import train, evaluate


# --------------------------------------------------------------------------- #
# Fine-tuning configurations: which layer groups are updated on the target
# subject's data (everything else is frozen). These map to the manuscript's
# Table II / Table III; the exact set is finalised in the experiment phase.
# --------------------------------------------------------------------------- #

#: group name -> attribute names on the EEGNet model
_GROUP_ATTRS = {
    "conv1": ["conv1"],
    "depthwise": ["depthwise_conv"],
    "separable_conv": ["separable_conv"],
    "dense": ["dense"],
    "electrode_se": ["se_electrode1", "se_electrode2"],
    "filter_se": ["se1", "se2", "se3"],
}

def _is_multiband(config) -> bool:
    """FBCNet consumes the filter-bank tensor; every other model is single-band."""
    return config.model.lower() == "fbcnet"


FINETUNE_CONFIGS: dict[str, list[str]] = {
    "dense": ["dense"],
    "se_dense": ["electrode_se", "filter_se", "dense"],
    "se_conv_dense": ["electrode_se", "filter_se", "separable_conv", "dense"],
    "electrode_se_dense": ["electrode_se", "dense"],
    "continued": ["__all__"],  # full fine-tuning (every parameter trainable)
}


def apply_finetune_config(model, groups) -> int:
    """Freeze everything, then unfreeze the requested layer groups.

    ``groups`` may be a preset name from :data:`FINETUNE_CONFIGS` or a list of
    group keys from :data:`_GROUP_ATTRS`. Returns the number of trainable
    parameters (used for Table III).
    """
    if isinstance(groups, str):
        groups = FINETUNE_CONFIGS[groups]

    for p in model.parameters():
        p.requires_grad = False

    if "__all__" in groups:
        for p in model.parameters():
            p.requires_grad = True
    else:
        for group in groups:
            for attr in _GROUP_ATTRS[group]:
                module = getattr(model, attr, None)
                if module is None:
                    continue  # group absent in this configuration (e.g. no SE)
                for p in module.parameters():
                    p.requires_grad = True

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- #
# Stage 1 -- base pre-training
# --------------------------------------------------------------------------- #

def pretrain_base(config, device, subjects=None, verbose: bool = False):
    """Train the subject-independent base model on pooled calibration trials."""
    X, Y = (calibration_dataset(config.data_dir)
            if subjects is None else calibration_dataset(config.data_dir, subjects))

    X_tr, X_val, Y_tr, Y_val = train_test_split(
        X, Y, train_size=1 - config.val_ratio,
        random_state=config.split_seed, shuffle=True, stratify=Y,
    )
    mb = _is_multiband(config)
    train_loader = make_loader(X_tr, Y_tr, config, shuffle=True, multiband=mb)
    val_loader = make_loader(X_val, Y_val, config, shuffle=False, multiband=mb)

    model = build_model(config).to(device)
    model = train(
        model, train_loader, val_loader, config, device,
        warmup_epochs=config.pretrain_warmup_epochs,
        min_acc=config.pretrain_min_acc,
        verbose=verbose,
    )
    return model


# --------------------------------------------------------------------------- #
# Stage 2 -- subject-specific fine-tuning and evaluation
# --------------------------------------------------------------------------- #

def finetune_subject(config, device, subject, base_state, finetune_config="se_dense", verbose=False):
    """Fine-tune a copy of the base model on one subject's calibration trials.

    Parameters
    ----------
    base_state : dict
        ``state_dict`` of the pre-trained base model.
    finetune_config : str or list
        Which layer groups to update (see :data:`FINETUNE_CONFIGS`).

    Returns
    -------
    (model, n_trainable)
    """
    X_calib, Y_calib, _, _ = online_session(config.data_dir, subject)
    X_tr, X_val, Y_tr, Y_val = train_test_split(
        X_calib, Y_calib, train_size=1 - config.val_ratio,
        random_state=config.split_seed, shuffle=True, stratify=Y_calib,
    )
    mb = _is_multiband(config)
    train_loader = make_loader(X_tr, Y_tr, config, shuffle=True, multiband=mb)
    val_loader = make_loader(X_val, Y_val, config, shuffle=False, multiband=mb)

    model = build_model(config).to(device)
    model.load_state_dict(copy.deepcopy(base_state))
    n_trainable = apply_finetune_config(model, finetune_config)

    model = train(model, train_loader, val_loader, config, device, verbose=verbose)
    return model, n_trainable


def evaluate_online(model, config, device, subject, *, return_scales=False):
    """Evaluate ``model`` on a subject's held-out online session."""
    _, _, X_online, Y_online = online_session(config.data_dir, subject)
    loader = make_loader(X_online, Y_online, config, shuffle=False, multiband=_is_multiband(config))
    return evaluate(model, loader, device, return_scales=return_scales)
