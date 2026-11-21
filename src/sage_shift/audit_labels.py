"""SAGE audit section 4: label strings, per-class counts, co-occurrence and the `class` column."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from sage_shift.audit_types import BLANK, SPLIT_KEYS, Audit, SageData, by_split, count_by_split, pct
from sage_shift.data import SAGE_CLASSES, SAGE_LABELS, SPLITS, label_indicators, parse_label_list


def _normalise(label: str) -> str:
    """Case-, space- and underscore-insensitive form, used only to spot spelling variants."""
    return re.sub(r"[\s_]+", "-", label.strip().lower())


def _raw_label_table(label_lists: pd.Series) -> pd.DataFrame:
    """Every raw label string with its count, whether the README lists it, and likely variants."""
    counts = Counter(label for items in label_lists for label in items)
    known = set(SAGE_LABELS)
    spellings: dict[str, list[str]] = {}
    for raw in counts:
        spellings.setdefault(_normalise(raw), []).append(raw)
    rows = []
    for raw, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        normalised = _normalise(raw)
        others = [other for other in spellings[normalised] if other != raw]
        variant_of = normalised if raw not in known and normalised in known else (others[0] if others else "")
        rows.append(
            {
                "raw_label": raw,
                "n_images": count,
                "in_readme_list": raw in known,
                "possible_variant_of": variant_of,
            }
        )
    return pd.DataFrame(rows, columns=["raw_label", "n_images", "in_readme_list", "possible_variant_of"])


def _parse_all_labels(cells: pd.Series) -> tuple[pd.Series, int]:
    """Parse every `labels` cell; unparseable cells become None and are counted."""
    parsed: list[list[str] | None] = []
    failures = 0
    for cell in cells:
        try:
            parsed.append(parse_label_list(cell))
        except (ValueError, SyntaxError, TypeError):
            parsed.append(None)
            failures += 1
    return pd.Series(parsed, index=cells.index, dtype=object), failures


def audit_labels(data: SageData, audit: Audit, cfg: dict[str, Any]) -> pd.DataFrame:
    """Section 4: raw label strings, per-class counts, co-occurrence and the Figure 3 comparison.

    Returns samples.csv joined with classification.csv, plus a parsed `label_list` column.
    """
    section = "4 Labels"
    table = data.samples.merge(data.classification, on="sample_uuid", validate="one_to_one")
    label_lists, failures = _parse_all_labels(table["labels"])
    audit.check(
        section,
        "every `labels` cell parses as a list of strings",
        failures == 0,
        f"{failures} of {len(table)} cells failed",
    )
    if failures:
        raise ValueError(f"{failures} `labels` cells could not be parsed; the label audit cannot continue")
    table["label_list"] = label_lists

    raw_table = _raw_label_table(table["label_list"])
    audit.tables["labels_raw_strings"] = raw_table
    observed = set(raw_table["raw_label"])
    unknown = sorted(observed - set(SAGE_LABELS))
    absent = sorted(set(SAGE_LABELS) - observed)
    variants = raw_table.loc[raw_table["possible_variant_of"] != "", "raw_label"].tolist()
    audit.check(
        section,
        f"raw label strings are exactly the {len(SAGE_LABELS)} README labels",
        not unknown and not absent and not variants,
        f"{len(observed)} distinct strings; unknown: {unknown or 'none'}; "
        f"absent: {absent or 'none'}; spelling variants: {variants or 'none'}",
    )
    repeated = sum(len(items) != len(set(items)) for items in table["label_list"])
    audit.check(section, "no label is repeated within a row", repeated == 0, f"{repeated} rows repeat a label")

    label_space = [*SAGE_LABELS, *unknown]  # unknown strings are kept as extra columns, never dropped
    indicators = label_indicators(table["label_list"], label_space)
    patient_any = indicators.groupby(table["patient_uuid"]).max()
    patient_split = table.groupby("patient_uuid")["split"].first()
    thresholds = cfg["thresholds"]
    rows = []
    for label in label_space:
        row: dict[str, Any] = {"label": label}
        for key in SPLIT_KEYS:
            images = indicators[label] if key == "total" else indicators.loc[table["split"] == key, label]
            patients = patient_any[label] if key == "total" else patient_any.loc[patient_split == key, label]
            row[f"{key}_images"] = int(images.sum())
            row[f"{key}_pct"] = pct(images.sum(), len(images))
            row[f"{key}_patients"] = int(patients.sum())
        row["low_test_support"] = (
            row["test_images"] < thresholds["min_test_images"]
            or row["test_patients"] < thresholds["min_test_patients"]
        )
        rows.append(row)
    label_counts = pd.DataFrame(rows).sort_values(["total_images", "label"], ascending=[False, True])
    label_counts = label_counts.reset_index(drop=True)
    audit.tables["label_counts"] = label_counts
    order = list(label_counts["label"])

    matrix = indicators[order].to_numpy(dtype=np.int64)
    cooccurrence = pd.DataFrame(matrix.T @ matrix, index=order, columns=order)
    cooccurrence.index.name = "label"
    audit.tables["label_cooccurrence"] = cooccurrence

    n_labels = indicators.sum(axis=1)
    audit.tables["labels_per_image"] = (
        count_by_split(n_labels, table["split"]).rename_axis("n_labels").reset_index()
    )

    paper = cfg["paper_figure3_counts"]
    figure3 = pd.DataFrame(
        {
            "label": order,
            "file_count": list(label_counts["total_images"]),
            "paper_count": [paper.get(label) for label in order],
        }
    )
    figure3["difference"] = figure3["file_count"] - figure3["paper_count"]
    audit.tables["figure3_comparison"] = figure3
    mismatched = figure3[figure3["difference"] != 0]
    mismatch_text = "; ".join(
        f"{row.label}: file {row.file_count} vs paper {row.paper_count} ({row.difference:+.0f})"
        for row in mismatched.itertuples()
    )
    only_in_paper = sorted(set(paper) - set(order))
    audit.check(
        section,
        "per-class totals match the paper's Figure 3",
        mismatched.empty and not only_in_paper,
        mismatch_text or "all classes match",
    )
    if not mismatched.empty:
        audit.discrepancies.append(
            f"Figure 3 class counts: {mismatch_text}. All other classes match. "
            f"File total is {int(figure3['file_count'].sum())} label instances; the paper's bars sum to "
            f"{int(figure3['paper_count'].sum())}."
        )

    flagged = label_counts[label_counts["low_test_support"]]
    audit.check(
        section,
        f"every class has at least {thresholds['min_test_images']} test images and "
        f"{thresholds['min_test_patients']} test patients",
        flagged.empty,
        "; ".join(
            f"{row.label}: {row.test_images} images, {row.test_patients} patients" for row in flagged.itertuples()
        )
        or "no class below the thresholds",
    )
    audit.summary["labels"] = {
        "n_labels": len(label_space),
        "unknown_label_strings": unknown,
        "label_instances": int(matrix.sum()),
        "labels_per_image_mean": by_split(
            table.assign(n=n_labels), lambda frame: round(float(frame["n"].mean()), 2)
        ),
        "labels_per_image_max": int(n_labels.max()),
        "zero_label_images": by_split(table.assign(n=n_labels), lambda frame: int((frame["n"] == 0).sum())),
        "largest_class": {"label": order[0], "n": int(label_counts["total_images"].iloc[0])},
        "smallest_class": {"label": order[-1], "n": int(label_counts["total_images"].iloc[-1])},
        "low_test_support": flagged["label"].tolist(),
        "figure3_mismatches": mismatched["label"].tolist(),
    }
    return table


def _strip_suffix(name: str, suffixes: tuple[str, ...]) -> str:
    """Remove a trailing class-name suffix such as '-normal'."""
    for suffix in suffixes:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _group_signature(items: list[str], group_of: dict[str, str], group_order: list[str]) -> str:
    """Which label groups a row's labels fall in, e.g. 'anatomy + pathology'."""
    if not items:
        return "no labels"
    present = [g for g in group_order if any(group_of.get(label, "ungrouped") == g for label in items)]
    if len(present) == 1:
        return f"{present[0]} only, {'1 label' if len(items) == 1 else '2+ labels'}"
    return " + ".join(present)


def _blank_vs_classed(category: pd.Series, blank: pd.Series, split: pd.Series) -> pd.DataFrame:
    """Rows per category, divided into blank-`class` rows (per split) and classed rows."""
    frame = pd.DataFrame({"category": category, "blank": blank, "split": split})
    out = frame.groupby("category").agg(n_rows=("blank", "size"), blank_total=("blank", "sum"))
    for name in SPLITS:
        out[f"blank_{name}"] = frame[frame["blank"] & (frame["split"] == name)].groupby("category").size()
    out = out.fillna(0).astype(int)
    out["classed_total"] = out["n_rows"] - out["blank_total"]
    return out


def audit_class_column(table: pd.DataFrame, audit: Audit, cfg: dict[str, Any]) -> None:
    """Section 4 (continued): how the `class` column relates to the multi-label `labels` column."""
    section = "4 Labels"
    suffixes = tuple(cfg["class_suffixes"])
    group_of = {label: group for group, labels in cfg["label_groups"].items() for label in labels}
    group_order = [*cfg["label_groups"], "ungrouped"]

    blank = table["class"].isna()
    classed = table[~blank]
    observed = sorted(classed["class"].unique())
    base_of = {name: _strip_suffix(name, suffixes) for name in observed}
    classable = set(base_of.values()) & set(SAGE_LABELS)
    audit.check(
        section,
        f"`class` holds exactly the {len(SAGE_CLASSES)} README class names (or is blank)",
        set(observed) == set(SAGE_CLASSES),
        f"{len(observed)} distinct non-blank values; {int(blank.sum())} blank rows",
    )

    class_counts = count_by_split(table["class"].fillna(BLANK), table["split"])
    label_set = table["label_list"].map(lambda items: ", ".join(sorted(items)) or "(none)")
    exact_hit = pd.Series(
        [name in items for name, items in zip(classed["class"], classed["label_list"])], index=classed.index
    )
    base_hit = pd.Series(
        [base_of[name] in items for name, items in zip(classed["class"], classed["label_list"])],
        index=classed.index,
    )
    rows = []
    for name in observed:
        members = classed.index[classed["class"] == name]
        top_set, top_n = Counter(label_set[members]).most_common(1)[0]
        rows.append(
            {
                "class": name,
                "base_label": base_of[name],
                **class_counts.loc[name].to_dict(),
                "base_label_in_row_labels": int(base_hit[members].sum()),
                "most_common_label_set": top_set,
                "most_common_label_set_n": top_n,
            }
        )
    if blank.any():
        rows.append({"class": BLANK, "base_label": "", **class_counts.loc[BLANK].to_dict()})
    audit.tables["class_counts"] = pd.DataFrame(rows)

    n_any = table["label_list"].map(len)
    n_classable = table["label_list"].map(lambda items: len(set(items) & classable))
    profile = pd.Series("2+ classable labels", index=table.index)
    profile[n_classable == 1] = "exactly 1 classable label"
    profile[(n_classable == 0) & (n_any > 0)] = "only labels with no class counterpart"
    profile[n_any == 0] = "no labels"
    audit.tables["class_blank_by_classable_labels"] = _blank_vs_classed(profile, blank, table["split"]).reset_index()

    signature = table["label_list"].map(lambda items: _group_signature(items, group_of, group_order))
    class_group = classed["class"].map(lambda name: group_of.get(base_of[name], "ungrouped"))
    winners = pd.crosstab(signature[classed.index], class_group).add_prefix("classed_as_")
    by_groups = _blank_vs_classed(signature, blank, table["split"]).join(winners).fillna(0).astype(int)
    audit.tables["class_blank_by_label_groups"] = (
        by_groups.sort_values("n_rows", ascending=False).rename_axis("labels_fall_in").reset_index()
    )
    audit.tables["class_blank_top_label_sets"] = (
        count_by_split(label_set[blank], table.loc[blank, "split"])
        .sort_values("total", ascending=False)
        .head(15)
        .rename_axis("label_set")
        .reset_index()
    )

    audit.check(
        section,
        "`class` (suffix removed) is always one of the row's labels",
        bool(base_hit.all()),
        f"{int(base_hit.sum())} of {len(classed)} classed rows; exact string match in {int(exact_hit.sum())}",
    )

    # Which kinds of labels accompany suffixed class names such as '-normal'.
    suffix_of = classed["class"].map(lambda name: next((s for s in suffixes if name.endswith(s)), "(no suffix)"))
    suffix_rows = []
    for suffix in [*suffixes, "(no suffix)"]:
        members = classed.loc[suffix_of == suffix, "label_list"]
        row: dict[str, Any] = {"class_suffix": suffix, "n_rows": len(members)}
        for group in cfg["label_groups"]:
            row[f"rows_with_{group}_label"] = sum(any(group_of.get(l) == group for l in items) for items in members)
        suffix_rows.append(row)
    audit.tables["class_suffix_profile"] = pd.DataFrame(suffix_rows)

    zero_classable = n_classable == 0
    blank_classable_groups = Counter(
        _group_signature([label for label in items if label in classable], group_of, group_order)
        for items in table.loc[blank & ~zero_classable, "label_list"]
    )
    audit.summary["class_column"] = {
        "n_values": len(observed),
        "n_classed": by_split(table, lambda frame: int(frame["class"].notna().sum())),
        "n_blank": by_split(table, lambda frame: int(frame["class"].isna().sum())),
        "suffixes_removed_for_comparison": list(suffixes),
        "class_exactly_in_labels": int(exact_hit.sum()),
        "class_base_in_labels": int(base_hit.sum()),
        "labels_without_class": [label for label in SAGE_LABELS if label not in classable],
        "rows_without_classable_label": {
            "blank": int((zero_classable & blank).sum()),
            "classed": int((zero_classable & ~blank).sum()),
        },
        "blank_rows_with_no_labels": int((blank & (n_any == 0)).sum()),
        "blank_rows_with_classable_label": int((blank & ~zero_classable).sum()),
        "blank_rows_with_classable_label_by_groups": dict(blank_classable_groups.most_common()),
        "classed_rows_with_2plus_classable_labels": int((~blank & (n_classable >= 2)).sum()),
        "suffix_profile": suffix_rows,
    }
