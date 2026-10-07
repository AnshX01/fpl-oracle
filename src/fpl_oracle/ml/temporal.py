"""Chronological expanding folds with indivisible season/GW blocks."""

import numpy as np


def temporal_folds(X, n_splits=3):
    meta = X.attrs.get("meta")
    if meta is None or "season" not in meta or "round" not in meta:
        raise ValueError("Temporal calibration requires aligned season/round metadata")
    meta = (
        meta.loc[X.index]
        if X.index.is_unique and meta.index.is_unique and all(i in meta.index for i in X.index)
        else meta
    )
    if len(meta) != len(X):
        raise ValueError("Temporal calibration metadata length mismatch")
    groups = list(zip(meta["season"].astype(str), meta["round"].astype(int), strict=True))
    ordered = sorted(set(groups))
    if len(ordered) < n_splits + 1:
        raise ValueError("Insufficient chronological gameweeks for calibration")
    blocks = np.array_split(np.arange(len(ordered)), n_splits + 1)
    for i in range(1, len(blocks)):
        cutoff = ordered[int(blocks[i][0])]
        upper = ordered[int(blocks[i][-1])]
        train = np.array([j for j, group in enumerate(groups) if group < cutoff])
        test = np.array([j for j, group in enumerate(groups) if cutoff <= group <= upper])
        if len(train) and len(test):
            yield train, test
