"""Audit the SAGE dataset: schema, integrity, split leakage, labels, metadata and text.

Read-only on data/ and makes no network calls. Writes to outputs/audit/ (git-ignored).
Run from the project root with the package installed (`pip install -e .`):

    python scripts/01_audit_sage.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sage_shift import audit_figures, audit_report
from sage_shift.audit import run_audit
from sage_shift.data import SPLITS, SagePaths, read_json

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the SAGE dataset (read-only, no network).")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "sage_audit.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "audit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = read_json(args.config)
    paths = SagePaths(args.data_dir)
    if args.out_dir.resolve().is_relative_to(paths.root.resolve()):
        raise SystemExit("Refusing to write audit outputs inside the data directory (data/ is read-only).")

    audit, table = run_audit(paths, cfg, on_schema=lambda a: print(audit_report.schema_text(a), flush=True))

    figures_dir = args.out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    n_images = {key: value["images"] for key, value in audit.summary["splits"]["counts"].items()}
    audit_figures.plot_class_bars(
        audit.tables["label_counts"], n_images, figures_dir / "class_counts_train_vs_test.png"
    )
    audit_figures.plot_cooccurrence(audit.tables["label_cooccurrence"], figures_dir / "label_cooccurrence.png")
    for split in SPLITS:
        chosen = audit_figures.pick_grid_samples(table, split, cfg["seed"])
        audit_figures.plot_sample_grid(
            [paths.image_path(row.patient_uuid, row.visit_uuid, row.sample_uuid) for row in chosen.itertuples()],
            [", ".join(items) or "(no labels)" for items in chosen["label_list"]],
            f"SAGE {split} split: {len(chosen)} random images (seed {cfg['seed']})",
            figures_dir / f"sample_grid_{split}.png",
        )

    report_path = audit_report.write_outputs(audit, cfg, args.out_dir)
    print("=== All checks ===")
    print(audit_report.checks_text(audit))
    print(f"\nReport:  {report_path}")
    print(f"Tables:  {args.out_dir / 'tables'}")
    print(f"Figures: {figures_dir}")
    if audit.hard_failures:
        print(f"\nHARD FAIL: {len(audit.hard_failures)} leakage check(s) failed.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
