"""Loading the per-subject ``.mat`` trial files and assembling datasets.

Each ``S{nn}_mitrials.mat`` holds ``Xtrain``/``Ytrain`` (calibration session)
and, for the online subjects, ``Xtest``/``Ytest``/``Ytest_fb`` (online session;
``Ytest_fb`` is the label the closed-loop feedback actually presented). Trials
are shaped ``(n_trials, n_samples, n_channels)``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io

from eegnetse.config import CALIBRATION_ONLY_SUBJECTS, ONLINE_SUBJECTS


def _subject_path(data_dir: Path, subject: int) -> Path:
    return Path(data_dir) / f"S{subject:02d}_mitrials.mat"


def load_subject(data_dir: Path, subject: int):
    """Load one subject's ``.mat`` file.

    Returns
    -------
    dict with keys ``Xtrain``, ``Ytrain`` and -- when an online session exists
    (i.e. the test arrays are not all-NaN) -- ``Xtest``, ``Ytest``, ``Ytest_fb``.
    """
    mat = scipy.io.loadmat(_subject_path(data_dir, subject))
    out = {"Xtrain": mat["Xtrain"], "Ytrain": mat["Ytrain"].squeeze()}

    Xte, Yte, Yte_fb = mat["Xtest"], mat["Ytest"], mat["Ytest_fb"]
    if not (np.isnan(Xte).all() or np.isnan(Yte).all() or np.isnan(Yte_fb).all()):
        out["Xtest"] = Xte
        out["Ytest"] = Yte.squeeze()
        out["Ytest_fb"] = Yte_fb.squeeze()
    return out


def calibration_dataset(data_dir: Path, subjects=CALIBRATION_ONLY_SUBJECTS):
    """Pool the calibration trials of ``subjects`` into one (X, Y) dataset.

    Defaults to S01-S07, the pre-training pool for the subject-independent base
    model (7 x 72 = 504 trials).
    """
    Xs, Ys = [], []
    for sub in subjects:
        data = load_subject(data_dir, sub)
        Xs.append(data["Xtrain"])
        Ys.append(data["Ytrain"])
    return np.concatenate(Xs, axis=0), np.concatenate(Ys, axis=0)


def online_session(data_dir: Path, subject: int):
    """Return ``(X_calib, Y_calib, X_online, Y_online)`` for an online subject."""
    data = load_subject(data_dir, subject)
    if "Xtest" not in data:
        raise ValueError(f"Subject S{subject:02d} has no online session.")
    return (
        np.asarray(data["Xtrain"]),
        np.asarray(data["Ytrain"]),
        np.asarray(data["Xtest"]),
        np.asarray(data["Ytest"]),
    )


def loso_split(data_dir: Path, target: int, subjects=ONLINE_SUBJECTS):
    """Leave-one-subject-out split for cross-validation (experiment A3).

    Returns
    -------
    (X_pool, Y_pool) : calibration trials of every subject except ``target``.
    (X_calib, Y_calib, X_online, Y_online) : the target subject's own sessions.
    """
    if target not in subjects:
        raise ValueError(f"Target S{target:02d} not in {subjects}.")
    others = [s for s in subjects if s != target]
    X_pool, Y_pool = calibration_dataset(data_dir, subjects=others)
    return (X_pool, Y_pool), online_session(data_dir, target)
