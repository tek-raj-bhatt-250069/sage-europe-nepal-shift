"""Write the SAGE audit outputs: CSV tables, JSON files and the Markdown report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from sage_shift.audit_types import SPLIT_KEYS, Audit, pct
from sage_shift.data import SPLITS


def _jsonable(value: Any) -> Any:
    """Convert numpy scalars and sets for json.dump."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError(f"cannot serialise {type(value).__name__}")


def _write_json(payload: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=_jsonable)
        handle.write("\n")


def write_outputs(audit: Audit, cfg: dict[str, Any], out_dir: Path) -> Path:
    """Write tables, audit_summary.json, schema_examples.json and AUDIT_REPORT.md; return the report path."""
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    for name, table in audit.tables.items():
        table.to_csv(tables_dir / f"{name}.csv", index=table.index.name is not None)
    summary = {
        "config": {key: cfg[key] for key in ("seed", "thresholds", "paper_tolerance_pct_points")},
        **audit.summary,
        "checks": [
            {"section": c.section, "check": c.name, "result": c.status, "hard": c.hard, "detail": c.detail}
            for c in audit.checks
        ],
        "discrepancies": audit.discrepancies,
    }
    _write_json(summary, out_dir / "audit_summary.json")
    _write_json(audit.examples, out_dir / "schema_examples.json")
    report_path = out_dir / "AUDIT_REPORT.md"
    report_path.write_text(render_report(audit, cfg), encoding="utf-8")
    return report_path


# --- Console text -----------------------------------------------------------------------------


def checks_text(audit: Audit, section: str | None = None) -> str:
    """One line per check, optionally limited to one section."""
    checks = [check for check in audit.checks if section is None or check.section == section]
    return "\n".join(
        f"[{check.status}]{' (hard)' if check.hard else ''} {check.name}: {check.detail}" for check in checks
    )


def mapping_lines(audit: Audit) -> list[str]:
    """How the IDs link the files and the image tree (verified by the section 1 and 2 checks)."""
    counts = audit.summary["counts"]
    return [
        f"patient_uuid ({counts['n_patients']}) > visit_uuid ({counts['n_visits']}) > "
        f"sample_uuid ({counts['n_images']}); each level nests inside the one before.",
        "samples.csv         one row per sample: patient_uuid, visit_uuid, sample_uuid, split",
        "classification.csv  one row per sample; joins to samples.csv on sample_uuid",
        "metadata.csv        one row per visit; joins to samples.csv on (patient_uuid, visit_uuid)",
        "captions.json       object keyed by sample_uuid",
        "vqa.json            object keyed by sample_uuid",
        "images              images/<patient_uuid>/<visit_uuid>/<sample_uuid>/<sample_uuid>.jpg",
    ]


def schema_text(audit: Audit) -> str:
    """Console text for section 1: per-file schema, example records and the ID mapping."""
    lines = ["", "=== 1. Schema discovery ==="]
    for name, info in audit.summary["files"].items():
        unit = "rows" if name.endswith(".csv") else "top-level keys"
        lines += ["", f"--- {name}: {info['n_rows']:,} {unit}"]
        if "structure" in info:
            lines.append(f"structure: {info['structure']}")
        lines.append(audit.tables[f"schema_{Path(name).stem}"].to_string(index=False))
        lines.append("examples: " + json.dumps(audit.examples[name], indent=2, ensure_ascii=False))
    lines += ["", "--- ID mapping", *mapping_lines(audit)]
    lines += ["", "--- Section 1 checks", checks_text(audit, section="1 Schema"), ""]
    return "\n".join(lines)


# --- Markdown report --------------------------------------------------------------------------


def _md_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    """A GitHub-flavoured Markdown table."""
    def cells(row: list[Any]) -> str:
        return "| " + " | ".join(str(cell).replace("|", "\\|") for cell in row) + " |"

    return [cells(headers), "|" + "|".join("---" for _ in headers) + "|", *(cells(row) for row in rows), ""]


def _group_text(stats: dict[str, Any]) -> str:
    return f"{stats['mean']:.2f} ± {stats['sd']:.2f} ({stats['min']}–{stats['max']})"


def _key_numbers(audit: Audit) -> list[str]:
    s = audit.summary
    counts = s["splits"]["counts"]
    sizes = s["splits"]["group_sizes"]
    n_images = {key: counts[key]["images"] for key in SPLIT_KEYS}

    def share(unit: str) -> list[str]:
        total = counts["total"][unit]
        return [f"{counts[k][unit]:,} ({pct(counts[k][unit], total)}%)" for k in SPLITS] + [f"{total:,}"]

    def of_images(values: dict[str, int]) -> list[str]:
        return [f"{values[k]:,} ({pct(values[k], n_images[k])}%)" for k in SPLIT_KEYS]

    meta, text = s["metadata"], s["text"]
    rows = [
        ["Images", *share("images")],
        ["Patients", *share("patients")],
        ["Visits", *share("visits")],
        *(
            [f"{name.capitalize()}, mean ± SD (min–max)", *(_group_text(sizes[name][k]) for k in SPLIT_KEYS)]
            for name in ("images per patient", "images per visit")
        ),
        ["Labels per image, mean", *(s["labels"]["labels_per_image_mean"][k] for k in SPLIT_KEYS)],
        ["Images with no label", *of_images(s["labels"]["zero_label_images"])],
        ["Images with blank `class`", *of_images(s["class_column"]["n_blank"])],
        [
            "Patients with age / with gender",
            *(f"{meta['pct_patients_with_age'][k]}% / {meta['pct_patients_with_gender'][k]}%" for k in SPLIT_KEYS),
        ],
        [
            "Images with a device recorded (excluding 'Unknown')",
            *(f"{meta['pct_images_with_device'][k]}% ({meta['pct_images_with_known_device'][k]}%)" for k in SPLIT_KEYS),
        ],
        ["Captions with ≥1 hallucinated span", *of_images(text["captions"]["captions with >=1 hallucinated span"])],
        ["VQA pairs", *(f"{text['n_vqa_pairs'][k]:,}" for k in SPLIT_KEYS)],
    ]
    lines = ["## Key numbers", "", *_md_table(["", "Train", "Test", "Total"], rows)]

    images = s["images"]
    properties = audit.tables["image_properties"]
    property_text = "; ".join(
        f"{row.total:,} at {row.resolution} ({row.mode} {row.format})" for row in properties.itertuples()
    )
    other_files = sum(images["other_files_in_image_tree"].values())
    age = meta["age_patient_level"]["total"]
    gender_counts = sorted(meta["gender_patient_level"]["total"].items(), key=lambda item: item[0].startswith("("))
    gender = ", ".join(f"{name} {count}" for name, count in gender_counts)
    captions = text["captions"]
    lines += [
        "- **ID mapping:** " + mapping_lines(audit)[0] + " `samples.csv` and `classification.csv` have one row "
        "per sample, `metadata.csv` one row per visit, and `captions.json` / `vqa.json` are keyed by "
        "`sample_uuid`. Images sit at `images/<patient>/<visit>/<sample>/<sample>.jpg`.",
        f"- **Images:** {property_text}. File size median {images['file_size']['median_kb']} KB "
        f"({images['file_size']['min_kb']}–{images['file_size']['max_kb']}). "
        f"{images['duplicates_md5']['n_groups']} MD5 duplicate groups. {other_files:,} non-image Synapse manifest "
        "files in the image tree were ignored.",
        f"- **Age (patient level, n = {age['n']}):** mean {age['mean']}, median {age['median']}, range "
        f"{age['min']}–{age['max']}; {meta['patients_whose_age_differs_between_visits']} patients have different "
        f"ages across visits. **Gender (patients):** {gender}; "
        f"{meta['patients_with_conflicting_gender']} patients have visits that disagree on gender.",
        f"- **Device:** free text, {meta['n_device_raw_strings']} distinct raw strings "
        f"({meta['n_device_normalised_strings']} after collapsing whitespace and case); "
        f"{meta['n_visits_device_unknown']} visits say 'Unknown'.",
        f"- **Captions:** {captions['captions']['total']:,} captions; {captions['annotation spans']['total']:,} "
        f"annotation spans, of which {captions['hallucinated spans']['total']:,} hallucinated and "
        f"{captions['omission spans']['total']:,} omissions; "
        f"{captions['captions unchanged by the expert']['total']:,} captions unchanged by the expert.",
        "",
    ]
    return lines


def _checks(audit: Audit) -> list[str]:
    rows = [
        [check.section, check.name + (" (hard)" if check.hard else ""), f"**{check.status}**", check.detail]
        for check in audit.checks
    ]
    return ["## Checks", "", *_md_table(["Section", "Check", "Result", "Detail"], rows)]


def _labels(audit: Audit, cfg: dict[str, Any]) -> list[str]:
    figure3 = audit.tables["figure3_comparison"].set_index("label")
    rows = []
    for row in audit.tables["label_counts"].itertuples():
        paper = figure3.loc[row.label]
        rows.append(
            [
                row.label,
                f"{row.train_images} ({row.train_pct}%)",
                f"{row.test_images} ({row.test_pct}%)",
                row.total_images,
                f"{row.train_patients} / {row.test_patients} / {row.total_patients}",
                f"{paper['paper_count']:.0f}",
                f"{paper['difference']:+.0f}" if paper["difference"] else "0",
                "LOW" if row.low_test_support else "",
            ]
        )
    per_image = audit.tables["labels_per_image"]
    distribution = ", ".join(f"{row.n_labels}: {row.total}" for row in per_image.itertuples())
    thresholds = cfg["thresholds"]
    return [
        "## Labels",
        "",
        f"All {audit.summary['labels']['n_labels']} raw label strings match the README list exactly; none were "
        f"mapped or dropped. `LOW` marks classes with fewer than {thresholds['min_test_images']} test images or "
        f"{thresholds['min_test_patients']} test patients. Paper = Figure 3; Diff = file minus paper.",
        "",
        *_md_table(
            ["Label", "Train images", "Test images", "Total", "Patients train / test / total", "Paper", "Diff", "Flag"],
            rows,
        ),
        f"Images by number of labels (number of labels: images): {distribution}.",
        "",
    ]


def _class_column(audit: Audit) -> list[str]:
    info = audit.summary["class_column"]
    counts = audit.tables["class_counts"]
    rows = [[f"`{row['class']}`", row["train"], row["test"], row["total"]] for row in counts.to_dict("records")]
    groups = audit.tables["class_blank_by_label_groups"]
    group_rows = [
        [row.labels_fall_in, row.n_rows, row.blank_train, row.blank_test, row.blank_total, row.classed_total]
        for row in groups.itertuples()
    ]
    n_classed = info["n_classed"]["total"]
    no_classable = info["rows_without_classable_label"]
    suffixes = " / ".join(f"`{suffix}`" for suffix in info["suffixes_removed_for_comparison"])
    top_sets = audit.tables["class_blank_top_label_sets"].head(6)
    top_sets_text = "; ".join(f"{row.label_set} {row.total}" for row in top_sets.itertuples())
    by_groups = "; ".join(
        f"{name} {count}" for name, count in info["blank_rows_with_classable_label_by_groups"].items()
    )
    suffix_text = " ".join(
        f"Rows with a `{row['class_suffix']}` class ({row['n_rows']}) that carry a label from each group: "
        + ", ".join(
            f"{key.removeprefix('rows_with_').removesuffix('_label')} {value}"
            for key, value in row.items()
            if key.startswith("rows_with_")
        )
        + "."
        for row in info["suffix_profile"]
        if row["class_suffix"] in info["suffixes_removed_for_comparison"]
    )
    return [
        "## `class` column",
        "",
        *_md_table(["Class", "Train", "Test", "Total"], rows),
        f"- **Is `class` one of the row's labels?** As an exact string, in {info['class_exactly_in_labels']} of "
        f"{n_classed} classed rows, because five class names carry a suffix ({suffixes}). With the suffix removed, "
        f"in {info['class_base_in_labels']} of {n_classed}.",
        f"- **Most common label sets among blank rows:** {top_sets_text}.",
        "- **All rows by the kind of labels they carry** (the grouping of labels into anatomy / pathology / other "
        "is mine, for this table only):",
        "",
        *_md_table(
            ["Row's labels fall in", "Rows", "Blank train", "Blank test", "Blank total", "Classed"], group_rows
        ),
        f"- **Inference (not documented in the README):** `class` looks like a single-label view of `labels`. "
        f"{len(info['labels_without_class'])} labels have no class counterpart "
        f"({', '.join(info['labels_without_class'])}). Of the {no_classable['blank'] + no_classable['classed']} "
        f"rows whose labels are all among those, or empty, {no_classable['blank']} are blank. The other "
        f"{info['blank_rows_with_classable_label']} blank rows do carry a label that has a class (by the groups "
        f"of those labels: {by_groups}), but {info['classed_rows_with_2plus_classable_labels']} classed rows also "
        "carry two or more such labels, so the exclusion rule cannot be recovered from the labels alone. "
        f"{suffix_text}",
        "",
    ]


def _text(audit: Audit, cfg: dict[str, Any]) -> list[str]:
    text = audit.summary["text"]
    questions = audit.tables["vqa_questions"]
    rows = [
        [index, row.question, row.train, row.test, row.total, row.images_without_this_question]
        for index, row in enumerate(questions.itertuples(), start=1)
    ]
    lines = [
        "## Text annotations",
        "",
        f"The paper describes {cfg['expected']['n_vqa_questions_paper']} questions; `vqa.json` has "
        f"{text['n_question_templates']} distinct question templates. Pairs per image: "
        + ", ".join(f"{count} images with {n} pairs" for n, count in text["pairs_per_image"].items())
        + ".",
        "",
        *_md_table(["#", "Question template", "Train", "Test", "Total", "Images without it"], rows),
    ]
    visibility = text["visibility"]
    if not visibility["found"]:
        return [*lines, "No single visibility question was found.", ""]
    leading = audit.tables["vqa_visibility_leading_word"]
    leading_rows = [
        [row.word_after_stem, f"{row.train} ({row.train_pct}%)", f"{row.test} ({row.test_pct}%)", row.total]
        for row in leading.itertuples()
    ]
    return [
        *lines,
        f"**Visibility answers are free text:** {visibility['n_answers']['total']:,} answers, "
        f"{visibility['n_distinct_answers']} distinct. {visibility['n_answers_starting_with_stem']:,} start with "
        f"\"{visibility['answer_stem']} …\". The table shows the word that follows that stem, as written "
        "(percentages are of all images in the split). Nothing has been binned.",
        "",
        *_md_table(["Word after the stem", "Train", "Test", "Total"], leading_rows),
    ]


def _risks(audit: Audit) -> list[str]:
    s = audit.summary
    labels = audit.tables["label_counts"]
    weakest = labels.sort_values(["test_patients", "test_images"]).head(4)
    weakest_text = "; ".join(
        f"{row.label} {row.test_images} images / {row.test_patients} patients" for row in weakest.itertuples()
    )
    largest, smallest = s["labels"]["largest_class"], s["labels"]["smallest_class"]
    counts = s["splits"]["counts"]
    per_patient = s["splits"]["group_sizes"]["images per patient"]["total"]
    largest_test_patient = s["splits"]["group_sizes"]["images per patient"]["test"]["max"]
    zero = s["labels"]["zero_label_images"]["total"]
    blank = s["class_column"]["n_blank"]["total"]
    meta = s["metadata"]
    train_pct = s["splits"]["train_pct"]
    flagged = ", ".join(s["labels"]["low_test_support"]) or "none"
    risks = [
        f"**Small classes.** Weakest test support: {weakest_text}. Per-class test metrics for these will have "
        f"wide patient-level confidence intervals. Below the flag thresholds: {flagged}.",
        f"**Imbalance.** {largest['label']} has {largest['n']} images and {smallest['label']} {smallest['n']} "
        f"({largest['n'] / smallest['n']:.0f}:1), so accuracy-style metrics will be dominated by the large classes.",
        f"**Images with no label.** {zero} images ({pct(zero, counts['total']['images'])}%) carry no label. "
        "Decide whether they are valid all-negative examples before training and evaluation.",
        f"**Clustered data.** Images per patient range from {per_patient['min']} to {per_patient['max']}, "
        f"{s['splits']['patients_with_several_visits']} patients have several visits, and one test patient "
        f"contributes {largest_test_patient} of {counts['test']['images']} test images "
        f"({pct(largest_test_patient, counts['test']['images'])}%). Any validation split, "
        "cross-validation or patient-subset sampling (RQ3) must group by patient, and confidence intervals must "
        "resample patients, not images.",
        f"**Split ratio.** The official split is {train_pct['patients']}% train by patients and "
        f"{train_pct['images']}% by images (nominal {s['splits']['nominal_train_pct']}%).",
        f"**Missing metadata.** Age and gender exist for {meta['pct_patients_with_age']['total']}% of patients and "
        f"a device for {meta['pct_images_with_device']['total']}% of images "
        f"({meta['pct_images_with_known_device']['total']}% excluding 'Unknown'). Device is free text with "
        f"{meta['n_device_raw_strings']} spellings. Subgroup and device-stratified analyses will be small and need "
        "a manual device mapping first.",
        f"**`class` coverage.** `class` is blank for {blank} images ({pct(blank, counts['total']['images'])}%), so "
        "the 14-class task covers only part of the data; `labels` is the complete target.",
    ]
    if s["labels"]["figure3_mismatches"]:
        risks.append(
            f"**Paper mismatch.** Counts for {', '.join(s['labels']['figure3_mismatches'])} differ from Figure 3, so "
            "a reproduction may not match the paper's per-class numbers for those labels."
        )
    if s["images"]["off_size"]["n_images"]:
        risks.append(
            f"**Mixed image sizes.** {s['images']['off_size']['n_images']} images are not the standard size; fix "
            "the resize policy explicitly, and note it as a possible acquisition difference for RQ2."
        )
    if s["text"]["visibility"]["found"]:
        risks.append(
            f"**Image-quality strata.** Visibility answers are free text "
            f"({s['text']['visibility']['n_distinct_answers']} distinct answers) and need an agreed binning rule "
            "before they can be used as quality strata in RQ2."
        )
    return ["## Risks for later modelling", "", *(f"{i}. {risk}" for i, risk in enumerate(risks, start=1)), ""]


def render_report(audit: Audit, cfg: dict[str, Any]) -> str:
    """The Markdown audit report."""
    passed = sum(check.passed for check in audit.checks)
    hard = [check for check in audit.checks if check.hard]
    hard_status = "PASS" if all(check.passed for check in hard) else "FAIL"
    failed = "; ".join(check.name for check in audit.checks if not check.passed) or "none"
    lines = [
        "# SAGE data audit",
        "",
        f"Produced by `python scripts/01_audit_sage.py` from `data/files/` and `data/images/` (seed {cfg['seed']}, "
        "no network). Aggregates only: no patient-level rows. Tables are in `outputs/audit/tables/`, figures in "
        "`outputs/audit/figures/`.",
        "",
        f"**Result: {passed} of {len(audit.checks)} checks pass. Leakage checks (hard): {hard_status}.**",
        "",
        f"Failed checks: {failed}.",
        "",
        *_key_numbers(audit),
        *_checks(audit),
        *_labels(audit, cfg),
        *_class_column(audit),
        *_text(audit, cfg),
        "## Discrepancies with the paper or the README",
        "",
        *(f"{i}. {text}" for i, text in enumerate(audit.discrepancies, start=1)),
        "",
        *_risks(audit),
    ]
    return "\n".join(lines).rstrip() + "\n"
