"""Constant-predictor baselines on the SAGE six-class test set, compared with the paper's Table 6.

Reads the SAGE manifest written by scripts/03_build_manifests.py and uses only the test labels
(no model is involved). Prints and saves aggregate scores only. From the project root:

    python scripts/04_constant_baselines.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from sage_shift.baselines import closed_form_constant, constant_predictor_table, matching_constants
from sage_shift.data import read_json
from sage_shift.labelspace import SIX_CLASSES

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Constant-predictor baselines on the SAGE six-class test set.")
    parser.add_argument("--manifest", type=Path, default=ROOT / "outputs" / "manifests" / "sage_manifest.csv")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "experiment.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "baselines")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = read_json(args.config)
    manifest = pd.read_csv(args.manifest, dtype={"class": str}, keep_default_na=False)
    test = manifest[(manifest["in_six"] == 1) & (manifest["split"] == "test")]
    y_true = test["class"].tolist()
    table = constant_predictor_table(y_true, SIX_CLASSES)

    for row in table.itertuples():
        expected = closed_form_constant(row.support / len(y_true), len(SIX_CLASSES))
        for metric, value in expected.items():
            if abs(getattr(row, metric) - value) > 1e-12:
                raise SystemExit(f"closed form and scikit-learn disagree for {row.predict_always} / {metric}")

    print(f"Constant predictors on the SAGE six-class test set ({len(y_true)} images, "
          f"{test['patient_group'].nunique()} patients); scikit-learn scores, checked against the closed form:")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    best = float(table["weighted_f1"].max())
    comparisons = []
    print("\nPaper Table 6, HyperKvasir models evaluated on SAGE (weighted F1):")
    for model, scores in cfg["paper_table6"]["hyperkvasir"].items():
        matches = matching_constants(table, scores["sage"])
        verdict = (f"indistinguishable from always '{matches[0]}'" if matches
                   else "matches no constant predictor")
        relation = "below" if scores["sage"] < best else "at or above"
        print(f"  {model}: {scores['sage']:.4f} -> {verdict}; {relation} the best constant predictor ({best:.4f})")
        comparisons.append({"model": model, "paper_sage_weighted_f1": scores["sage"], "matching_constants": matches,
                            "below_best_constant": scores["sage"] < best})
    print("  (GastroVision rows are not compared: the authors' GastroVision label mapping is not in the released code.)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_dir / "constant_predictors.csv", index=False, float_format="%.6f")
    summary = {"test_images": len(y_true), "test_patients": int(test["patient_group"].nunique()),
               "constant_predictors": table.round(6).to_dict(orient="records"), "paper_comparison": comparisons}
    (args.out_dir / "baselines_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {args.out_dir / 'constant_predictors.csv'} and baselines_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
