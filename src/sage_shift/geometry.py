"""Geometry controls: inputs for fine-tuned models, bootstrap scores and the reading rules.

scripts/14_geometry_controls.py prepares the inputs and scores kept checkpoints;
scripts/15_geometry_analysis.py adds bootstrap intervals and applies the reading rules, which
were fixed before the runs.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from multiprocessing import Pool
from typing import Any

import numpy as np

from sage_shift.labelspace import SIX_CLASSES
from sage_shift.metrics import auroc, confusion, label_scores, macro_f1_many
from sage_shift.preprocess import (SAGEIFY, Placement, PreprocessReference, check_dataset, frame_field,
                                   geometry_images, load_rgb, sage_placement)

SAGE_INPUTS: tuple[str, ...] = ("raw", "masked", "masked_grey", "fov", "fov_grey", "letterbox", "letterbox_grey")
HK_INPUTS: tuple[str, ...] = ("raw", "sageify", "sageify_grey")
PAIRS: tuple[tuple[str, str], ...] = (("masked", "masked_grey"), ("letterbox", "letterbox_grey"), ("fov", "fov_grey"))
K = len(SIX_CLASSES)
POLYP = SIX_CLASSES.index("polyp")
LOWER_GI = (SIX_CLASSES.index("polyp"), SIX_CLASSES.index("rectum-retroflexion"))
RULES = {
    "collapse": "collapse = top class >= 80% of predictions (the re-run's collapse criterion)",
    "fill": "if masked collapses and masked_grey does not in at least two of three seeds, black area outside the "
            "field is sufficient to trigger the collapse, while if both collapse, scale, aspect or position is "
            "implicated instead; letterbox vs letterbox_grey and fov vs fov_grey are read the same way",
    "source_shortcut": "on HyperKvasir fold-1 upper-GI frames, a paired increase in polyp or rectum predictions "
                       "under sageify that is larger than under sageify_grey (interval of the difference excluding "
                       "0) shows a black-area shortcut on the source side",
}

_CONTEXT: dict[str, Any] = {}


# --- inputs for a fine-tuned model ---------------------------------------------------------------

def _set_context(reference: PreprocessReference, placement: Placement | None) -> None:
    _CONTEXT.update(reference=reference, placement=placement)


def _frame_inputs(task: tuple[str, str, int, tuple[str, ...]]) -> np.ndarray:
    path, dataset, size, names = task
    rgb = load_rgb(path)
    field = None if set(names) <= {"raw"} else frame_field(rgb, _CONTEXT["reference"], dataset).field
    images = geometry_images(rgb, field, size, names, _CONTEXT["placement"])
    return np.stack([images[name] for name in names])


def prepare_inputs(paths: Sequence[str], dataset: str, names: Sequence[str], reference: PreprocessReference,
                   size: int = 224, workers: int = 2) -> dict[str, np.ndarray]:
    """uint8 (N, size, size, 3) frames of each geometry input, in path order."""
    names = tuple(names)
    check_dataset(dataset, names)
    placement = sage_placement(reference) if SAGEIFY & set(names) else None
    tasks = [(str(path), dataset, size, names) for path in paths]
    out = {name: np.empty((len(tasks), size, size, 3), dtype=np.uint8) for name in names}

    def store(row: int, stack: np.ndarray) -> None:
        for k, name in enumerate(names):
            out[name][row] = stack[k]

    if workers <= 1:
        _set_context(reference, placement)
        for row, task in enumerate(tasks):
            store(row, _frame_inputs(task))
    else:
        with Pool(workers, initializer=_set_context, initargs=(reference, placement)) as pool:
            for row, stack in enumerate(pool.imap(_frame_inputs, tasks, chunksize=16)):
                store(row, stack)
    return out


# --- scores with bootstrap resamples (given as weights) ------------------------------------------

def class_shares(pred: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Share of predictions per class: (K,) without weights, (B, K) for a (B, N) weight matrix."""
    onehot = np.eye(K)[np.asarray(pred)]
    if weights is None:
        return onehot.mean(axis=0)
    counts = np.asarray(weights, dtype=float) @ onehot
    return counts / counts.sum(axis=1, keepdims=True)


def macro_auroc(truth: np.ndarray, proba: np.ndarray, weights: np.ndarray) -> float:
    """Mean one-vs-rest AUROC over the classes with both outcomes (as metrics.evaluate)."""
    values = [auroc(truth[:, c], proba[:, c], weights) for c in range(proba.shape[1])]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # no class with both outcomes gives nan
        return float(np.nanmean(values))


def sage_scores(y: np.ndarray, proba: np.ndarray, truth: np.ndarray,
                weights: np.ndarray) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Macro-F1, macro AUROC, polyp share and top-class share, at the point and on every resample."""
    pred = proba.argmax(axis=1)
    shares, boot_shares = class_shares(pred), class_shares(pred, weights)
    point = {"macro_f1": float(label_scores(confusion(y, pred, K))["macro_f1"]),
             "macro_auroc": macro_auroc(truth, proba, np.ones(len(y))),
             "polyp_share": float(shares[POLYP]), "top_class_share": float(shares.max()),
             "top_class": SIX_CLASSES[int(shares.argmax())]}
    boot = {"macro_f1": macro_f1_many(y, pred, K, weights),
            "macro_auroc": np.array([macro_auroc(truth, proba, w) for w in weights]),
            "polyp_share": boot_shares[:, POLYP], "top_class_share": boot_shares.max(axis=1)}
    return point, boot


def _ratio(numerator: np.ndarray, denominator: np.ndarray, weights: np.ndarray) -> tuple[float, np.ndarray]:
    """Weighted share numerator / denominator, at the point and on every resample (nan if empty)."""
    num, den = numerator.astype(float), denominator.astype(float)
    point = float(num.sum() / den.sum()) if den.sum() > 0 else float("nan")
    top, bottom = weights @ num, weights @ den
    return point, np.divide(top, bottom, out=np.full(len(weights), np.nan), where=bottom > 0)


def hk_scores(y: np.ndarray, pred: np.ndarray, upper: np.ndarray, hd: np.ndarray,
              weights: np.ndarray) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Per-class accuracy, macro-F1 and the share of upper-GI frames predicted polyp or rectum (all and HD)."""
    w = np.asarray(weights, dtype=float)
    lower = np.isin(pred, LOWER_GI)
    point: dict[str, float] = {}
    boot: dict[str, np.ndarray] = {}
    for c, name in enumerate(SIX_CLASSES):
        point[f"accuracy_{name}"], boot[f"accuracy_{name}"] = _ratio((y == c) & (pred == c), y == c, w)
    point["macro_f1"] = float(label_scores(confusion(y, pred, K))["macro_f1"])
    boot["macro_f1"] = macro_f1_many(y, pred, K, w)
    point["upper_gi_polyp_or_rectum_share"], boot["upper_gi_polyp_or_rectum_share"] = _ratio(upper & lower, upper, w)
    point["hd_upper_gi_polyp_or_rectum_share"], boot["hd_upper_gi_polyp_or_rectum_share"] = _ratio(
        upper & hd & lower, upper & hd, w)
    return point, boot


# --- reading rules ---------------------------------------------------------------------------------

def pair_verdict(black: Sequence[bool], grey: Sequence[bool], planned: int = 3, needed: int = 2) -> dict[str, Any]:
    """Fill rule on per-seed collapse flags of a black-fill input and its grey twin."""
    if len(black) != len(grey):
        raise ValueError("need one collapse flag per seed for both inputs")
    black_only = sum(bool(b) and not g for b, g in zip(black, grey))
    both = sum(bool(b) and bool(g) for b, g in zip(black, grey))
    left = max(0, planned - len(black))
    if black_only >= needed:
        verdict = "black area outside the field is sufficient to trigger the collapse"
    elif both >= needed:
        verdict = "scale, aspect or position is implicated instead"
    elif max(black_only, both) + left >= needed:
        verdict = f"undecided: {len(black)} of {planned} seeds available"
    else:
        verdict = "neither reading applies"
    return {"seeds": len(black), "black_collapses": int(sum(map(bool, black))),
            "grey_collapses": int(sum(map(bool, grey))), "black_only": black_only, "both": both, "verdict": verdict}


def shortcut_shown(increase: Mapping[str, float], difference: Mapping[str, float]) -> bool:
    """Source-side rule: an increase under sageify, larger than under sageify_grey (interval above 0)."""
    return bool(increase["estimate"] > 0 and difference["ci_low"] > 0)
