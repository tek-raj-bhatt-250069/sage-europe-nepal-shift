"""SAGE data audit: orchestration and the schema section.

Every check only reads from data/ and records aggregates on an `Audit` collector.
The other sections live in audit_integrity.py, audit_labels.py, audit_metadata.py and
audit_text.py; figures are in audit_figures.py and file writing in audit_report.py.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from sage_shift.audit_integrity import audit_integrity, audit_splits
from sage_shift.audit_labels import audit_class_column, audit_labels
from sage_shift.audit_metadata import audit_metadata
from sage_shift.audit_text import audit_text
from sage_shift.audit_types import ID_COLUMNS, Audit, SageData, load_sage
from sage_shift.data import SAGE_LABELS, SagePaths

PREVIEW_CHARS = 150


def run_audit(
    paths: SagePaths, cfg: dict[str, Any], on_schema: Callable[[Audit], None] | None = None
) -> tuple[Audit, pd.DataFrame]:
    """Run every audit section. Returns the collector and the per-sample table with parsed labels.

    `on_schema` is called right after the schema section, so the ID mapping can be
    reported before anything else is computed.
    """
    data = load_sage(paths)
    audit = Audit()
    audit_schema(data, audit, cfg["expected"])
    if on_schema is not None:
        on_schema(audit)
    audit_integrity(data, audit, cfg["expected"])
    audit_splits(data, audit, cfg["expected"])
    table = audit_labels(data, audit, cfg)
    audit_class_column(table, audit, cfg)
    audit_metadata(data, audit, cfg)
    audit_text(data, audit, cfg)
    return audit, table


def _preview(value: Any) -> Any:
    """Shorten long strings for example records."""
    if isinstance(value, str) and len(value) > PREVIEW_CHARS:
        return value[:PREVIEW_CHARS] + "…"
    return value


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Rows as plain dicts, with missing cells shown as None."""
    return [
        {key: (None if pd.isna(value) else value) for key, value in row.items()}
        for row in frame.to_dict("records")
    ]


def _csv_schema(path: Path) -> pd.DataFrame:
    """Column name, inferred dtype, missing count and distinct count for one CSV."""
    typed = pd.read_csv(path, keep_default_na=False, na_values=[""])
    return pd.DataFrame(
        {
            "column": list(typed.columns),
            "dtype": [str(dtype) for dtype in typed.dtypes],
            "n_missing": [int(typed[column].isna().sum()) for column in typed.columns],
            "n_distinct": [int(typed[column].nunique()) for column in typed.columns],
        }
    )


def _json_schema(records: Iterable[dict[str, Any]], prefix: str = "") -> pd.DataFrame:
    """Per key of a collection of dict records: value types, None count and empty-string count."""
    types: dict[str, Counter[str]] = {}
    n_none: Counter[str] = Counter()
    n_empty: Counter[str] = Counter()
    for record in records:
        for key, value in record.items():
            types.setdefault(key, Counter())[type(value).__name__] += 1
            n_none[key] += value is None
            n_empty[key] += isinstance(value, str) and value == ""
    return pd.DataFrame(
        [
            {
                "key": prefix + key,
                "types": ", ".join(f"{name} x{count}" for name, count in sorted(seen.items())),
                "n_records": sum(seen.values()),
                "n_none": n_none[key],
                "n_empty_string": n_empty[key],
            }
            for key, seen in types.items()
        ]
    )


def _text_examples(data: SageData) -> dict[str, Any]:
    """Three example records per JSON file, with long text truncated."""
    captions = [
        {
            "sample_uuid": sample_uuid,
            "gpt_description": _preview(caption.get("gpt_description")),
            "corrected_description": _preview(caption.get("corrected_description")),
            "annotation": f"{len(caption.get('annotation', []))} spans",
        }
        for sample_uuid, caption in list(data.captions.items())[:3]
    ]
    vqa = [
        {
            "sample_uuid": sample_uuid,
            "n_pairs": len(pairs),
            "first_pair": {key: _preview(value) for key, value in pairs[0].items()} if pairs else None,
        }
        for sample_uuid, pairs in list(data.vqa.items())[:3]
    ]
    return {"captions.json": captions, "vqa.json": vqa}


def audit_schema(data: SageData, audit: Audit, expected: dict[str, Any]) -> None:
    """Section 1: schema of the five annotation files and how their IDs link together."""
    section = "1 Schema"
    files_dir = data.paths.files_dir
    frames = {
        "classification.csv": data.classification,
        "samples.csv": data.samples,
        "metadata.csv": data.metadata,
    }
    files: dict[str, Any] = {}
    for name, frame in frames.items():
        audit.tables[f"schema_{Path(name).stem}"] = _csv_schema(files_dir / name)
        files[name] = {"n_rows": len(frame), "columns": list(frame.columns)}

    spans = [span for caption in data.captions.values() for span in caption.get("annotation", [])]
    pairs = [pair for items in data.vqa.values() for pair in items]
    audit.tables["schema_captions"] = pd.concat(
        [_json_schema(data.captions.values()), _json_schema(spans, prefix="annotation[].")],
        ignore_index=True,
    )
    audit.tables["schema_vqa"] = _json_schema(pairs, prefix="[].")
    files["captions.json"] = {
        "n_rows": len(data.captions),
        "structure": "object keyed by sample_uuid; value is an object with a list of annotation spans",
        "n_annotation_spans": len(spans),
    }
    files["vqa.json"] = {
        "n_rows": len(data.vqa),
        "structure": "object keyed by sample_uuid; value is a list of {question, answer}",
        "n_pairs": len(pairs),
    }
    audit.summary["files"] = files
    audit.examples = {
        "classification.csv": _records(data.classification.head(3)),
        "samples.csv": _records(data.samples.head(3)),
        "metadata.csv": "not shown: schema and aggregates only",
        **_text_examples(data),
    }

    samples = data.samples
    sample_ids = set(samples["sample_uuid"])
    visit_pairs = set(zip(samples["patient_uuid"], samples["visit_uuid"]))
    meta_pairs = list(zip(data.metadata["patient_uuid"], data.metadata["visit_uuid"]))
    counts = {
        "n_images": len(samples),
        "n_patients": int(samples["patient_uuid"].nunique()),
        "n_visits": int(samples["visit_uuid"].nunique()),
    }
    audit.summary["counts"] = counts

    audit.check(
        section,
        "sample_uuid is unique and IDs are never missing in samples.csv",
        samples["sample_uuid"].is_unique and bool(samples[ID_COLUMNS].notna().all().all()),
        f"{len(samples)} rows, {len(sample_ids)} distinct sample_uuid",
    )
    class_ids = data.classification["sample_uuid"]
    audit.check(
        section,
        "classification.csv matches samples.csv one-to-one on sample_uuid",
        class_ids.is_unique and set(class_ids) == sample_ids,
        f"{len(class_ids)} rows, {len(set(class_ids) ^ sample_ids)} IDs not shared",
    )
    for name, keyed in (("captions.json", data.captions), ("vqa.json", data.vqa)):
        audit.check(
            section,
            f"{name} keys are exactly the sample_uuid set",
            set(keyed) == sample_ids,
            f"{len(keyed)} keys, {len(set(keyed) ^ sample_ids)} not shared",
        )
    patients_per_visit = samples.groupby("visit_uuid")["patient_uuid"].nunique()
    audit.check(
        section,
        "every visit_uuid belongs to exactly one patient",
        bool((patients_per_visit == 1).all()),
        f"{int((patients_per_visit > 1).sum())} visits linked to several patients",
    )
    audit.check(
        section,
        "metadata.csv has one row per visit, matching the visits in samples.csv",
        len(meta_pairs) == len(set(meta_pairs)) and set(meta_pairs) == visit_pairs,
        f"{len(meta_pairs)} rows, {len(set(meta_pairs) ^ visit_pairs)} (patient, visit) pairs not shared",
    )
    wanted = {key: expected[key] for key in counts}
    audit.check(
        section,
        "image, patient and visit counts match the paper",
        counts == wanted,
        f"found {counts['n_images']} / {counts['n_patients']} / {counts['n_visits']}, "
        f"expected {wanted['n_images']} / {wanted['n_patients']} / {wanted['n_visits']}",
    )
    if not set(SAGE_LABELS) & set(data.classification.columns):
        audit.discrepancies.append(
            "Label storage: the README names a `labels` column but not its format; classification.csv has no "
            "per-label columns, and `labels` holds a stringified list per row, next to the `class` column."
        )
    if len(data.metadata) != counts["n_patients"]:
        audit.discrepancies.append(
            f"metadata.csv granularity: the README calls it patient-level, but it has {len(data.metadata)} rows, "
            f"one per visit, for {counts['n_patients']} patients."
        )
