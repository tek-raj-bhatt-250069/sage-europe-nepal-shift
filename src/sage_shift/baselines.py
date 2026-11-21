"""Scores of constant predictors (always the same class), computed as the SAGE authors computed F1.

The authors report scikit-learn's weighted F1 (`f1_score(..., average="weighted")`,
lib/trainer/utils.py:50 at commit e24eb0e). A class that is never predicted gets precision 0
here (`zero_division=0`), which equals scikit-learn's default value without the warning.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def constant_predictor_table(y_true: Sequence[str], classes: Sequence[str]) -> pd.DataFrame:
    """Weighted F1, macro F1 and accuracy of predicting each class for every image."""
    rows = []
    for cls in classes:
        y_pred = [cls] * len(y_true)
        rows.append({
            "predict_always": cls,
            "support": sum(1 for y in y_true if y == cls),
            "weighted_f1": f1_score(y_true, y_pred, labels=list(classes), average="weighted", zero_division=0),
            "macro_f1": f1_score(y_true, y_pred, labels=list(classes), average="macro", zero_division=0),
            "accuracy": accuracy_score(y_true, y_pred),
        })
    return pd.DataFrame(rows).sort_values("weighted_f1", ascending=False).reset_index(drop=True)


def closed_form_constant(prevalence: float, n_classes: int) -> dict[str, float]:
    """The same three scores from the class prevalence p: F1 of the chosen class is 2p / (1 + p)."""
    f1 = 2 * prevalence / (1 + prevalence)
    return {"weighted_f1": prevalence * f1, "macro_f1": f1 / n_classes, "accuracy": prevalence}


def matching_constants(table: pd.DataFrame, value: float, tolerance: float = 5e-5) -> list[str]:
    """Constant predictors whose weighted F1 equals `value` to within `tolerance` (default: 4-decimal rounding)."""
    return table.loc[(table["weighted_f1"] - value).abs() < tolerance, "predict_always"].tolist()
