"""SAGE audit section 5: demographic and device coverage (aggregates only)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from sage_shift.audit_types import BLANK, SPLIT_KEYS, Audit, SageData, by_split, count_by_split, pct


def _coverage_row(metric: str, frame: pd.DataFrame, flag: pd.Series) -> dict[str, Any]:
    """Count and percentage of rows where `flag` is true, per split and in total."""
    values = by_split(frame.assign(_flag=flag), lambda part: (int(part["_flag"].sum()), len(part)))
    row: dict[str, Any] = {"metric": metric}
    for key, (hits, size) in values.items():
        row.update({f"{key}_n": hits, f"{key}_of": size, f"{key}_pct": pct(hits, size)})
    return row


def _age_stats(ages: pd.Series) -> dict[str, float]:
    """n, mean, median, min and max of the recorded ages."""
    ages = ages.dropna()
    if ages.empty:
        return {"n": 0}
    return {
        "n": int(len(ages)),
        "mean": round(float(ages.mean()), 1),
        "median": round(float(ages.median()), 1),
        "min": round(float(ages.min()), 1),
        "max": round(float(ages.max()), 1),
    }


def audit_metadata(data: SageData, audit: Audit, cfg: dict[str, Any]) -> None:
    """Section 5: demographic and device coverage. Aggregates only; nothing per patient is stored."""
    section = "5 Metadata"
    meta = data.metadata.copy()
    meta["gender_norm"] = meta["gender"].str.strip().str.lower()
    meta["age_num"] = pd.to_numeric(meta["age"], errors="coerce")
    meta["device_norm"] = meta["device"].map(lambda value: " ".join(value.split()).lower(), na_action="ignore")
    meta["device_recorded"] = meta["device"].notna()
    meta["device_known"] = meta["device_recorded"] & ~meta["device_norm"].isin(cfg["device_unknown_values"])

    audit.tables["metadata_gender_raw_values"] = (
        meta["gender"].fillna(BLANK).value_counts().rename_axis("raw_value").reset_index(name="n_visits")
    )
    gender_variants = sorted(value for value in meta["gender"].dropna().unique() if value != value.strip().lower())
    audit.check(
        section,
        "gender uses one spelling per category",
        not gender_variants,
        f"variant spellings: {gender_variants or 'none'} "
        f"({int(meta['gender'].isin(gender_variants).sum())} visits); compared case-insensitively here",
    )
    bad_age = int((meta["age"].notna() & meta["age_num"].isna()).sum())
    audit.check(section, "age is numeric wherever it is recorded", bad_age == 0, f"{bad_age} non-numeric values")

    patients = meta.groupby("patient_uuid").agg(
        split=("split", "first"),
        gender=("gender_norm", "first"),
        n_gender_values=("gender_norm", "nunique"),
        n_visits_with_gender=("gender_norm", "count"),
        age=("age_num", "mean"),
        n_age_values=("age_num", "nunique"),
        n_visits=("visit_uuid", "size"),
    )
    conflicts = int((patients["n_gender_values"] > 1).sum())
    patients.loc[patients["n_gender_values"] > 1, "gender"] = "conflicting"
    audit.check(
        section,
        "no patient has visits that disagree on gender",
        conflicts == 0,
        f"{conflicts} patients with conflicting gender across visits",
    )

    images = data.samples[["sample_uuid", "visit_uuid", "split"]].merge(
        meta[["visit_uuid", "gender_norm", "age_num", "device_norm", "device_recorded", "device_known"]],
        on="visit_uuid",
        how="left",
        validate="many_to_one",
    )
    coverage = pd.DataFrame(
        [
            _coverage_row("patients with age (any visit)", patients, patients["age"].notna()),
            _coverage_row("patients with gender (any visit)", patients, patients["gender"].notna()),
            _coverage_row("visits with age", meta, meta["age_num"].notna()),
            _coverage_row("visits with gender", meta, meta["gender_norm"].notna()),
            _coverage_row("visits with device recorded", meta, meta["device_recorded"]),
            _coverage_row("visits with device recorded and not 'Unknown'", meta, meta["device_known"]),
            _coverage_row("images whose visit has age", images, images["age_num"].notna()),
            _coverage_row("images whose visit has gender", images, images["gender_norm"].notna()),
            _coverage_row("images with device recorded", images, images["device_recorded"].eq(True)),
            _coverage_row("images with device recorded and not 'Unknown'", images, images["device_known"].eq(True)),
        ]
    )
    audit.tables["metadata_coverage"] = coverage

    age_rows = [
        {"level": level, "split": key, **stats}
        for level, frame, column in (
            ("patient (mean over the patient's visits)", patients, "age"),
            ("visit", meta, "age_num"),
        )
        for key, stats in by_split(frame, lambda part, column=column: _age_stats(part[column])).items()
    ]
    audit.tables["metadata_age_stats"] = pd.DataFrame(age_rows)
    gender_counts = count_by_split(patients["gender"].fillna("(not recorded)"), patients["split"])
    audit.tables["metadata_gender_counts"] = gender_counts.rename_axis("gender_patient_level").reset_index()

    device_key = meta["device_norm"].fillna(BLANK)
    devices = count_by_split(device_key, meta["split"]).add_prefix("visits_")
    devices = devices.join(count_by_split(images["device_norm"].fillna(BLANK), images["split"]).add_prefix("images_"))
    devices["n_raw_spellings"] = meta.groupby(device_key)["device"].nunique()
    audit.tables["metadata_device_values"] = (
        devices.sort_values("visits_total", ascending=False).rename_axis("device_normalised").reset_index()
    )

    by_metric = coverage.set_index("metric")
    tolerance = cfg["paper_tolerance_pct_points"]
    expected = cfg["expected"]
    demo_pct = float(by_metric.loc["patients with age (any visit)", "total_pct"])
    gender_pct = float(by_metric.loc["patients with gender (any visit)", "total_pct"])
    device_pct = float(by_metric.loc["images with device recorded", "total_pct"])
    device_known_pct = float(by_metric.loc["images with device recorded and not 'Unknown'", "total_pct"])
    paper_demo = expected["patients_with_demographics_pct_paper"]
    paper_device = expected["images_with_device_pct_paper"]
    demo_ok = abs(demo_pct - paper_demo) <= tolerance and abs(gender_pct - paper_demo) <= tolerance
    # 'Unknown' is not a recorded device, so the paper figure is compared with the known-device share.
    device_ok = abs(device_known_pct - paper_device) <= tolerance
    audit.check(
        section,
        f"patients with age and gender is about {paper_demo}% (paper), within {tolerance} points",
        demo_ok,
        f"age {demo_pct}%, gender {gender_pct}% of patients",
    )
    audit.check(
        section,
        f"images with a known device is about {paper_device}% (paper), within {tolerance} points",
        device_ok,
        f"{device_known_pct}% of images have a device other than 'Unknown'; {device_pct}% have any non-blank value",
    )
    if not demo_ok:
        visits_with_age = int(by_metric.loc["visits with age", "total_n"])
        visits_over_patients = pct(visits_with_age, len(patients))
        text = (
            f"Demographic coverage: {demo_pct}% of patients have age and {gender_pct}% have gender; the paper says "
            f"about {paper_demo}%. By visits it is {by_metric.loc['visits with age', 'total_pct']}% and by images "
            f"{by_metric.loc['images whose visit has age', 'total_pct']}%."
        )
        if abs(visits_over_patients - paper_demo) <= tolerance:
            text += (
                f" Inference: {visits_with_age} visits have age, and {visits_with_age} / {len(patients)} patients "
                f"= {visits_over_patients}%, so the paper may have counted visit records against the patient total."
            )
        audit.discrepancies.append(text)
    if not device_ok:
        audit.discrepancies.append(
            f"Device coverage: {device_known_pct}% of images have a device other than 'Unknown' ({device_pct}% "
            f"have any non-blank value); the paper says about {paper_device}%."
        )

    def per_split_pct(metric: str) -> dict[str, float]:
        return {key: float(by_metric.loc[metric, f"{key}_pct"]) for key in SPLIT_KEYS}

    audit.summary["metadata"] = {
        "rows_are": "one per visit",
        "gender_variant_spellings": gender_variants,
        "patients_with_conflicting_gender": conflicts,
        "patients_with_gender_in_some_visits_only": int(
            ((patients["n_visits_with_gender"] > 0) & (patients["n_visits_with_gender"] < patients["n_visits"])).sum()
        ),
        "patients_whose_age_differs_between_visits": int((patients["n_age_values"] > 1).sum()),
        "pct_patients_with_age": per_split_pct("patients with age (any visit)"),
        "pct_patients_with_gender": per_split_pct("patients with gender (any visit)"),
        "pct_images_with_device": per_split_pct("images with device recorded"),
        "pct_images_with_known_device": per_split_pct("images with device recorded and not 'Unknown'"),
        "age_patient_level": by_split(patients, lambda part: _age_stats(part["age"])),
        "gender_patient_level": {key: gender_counts[key].to_dict() for key in SPLIT_KEYS},
        "n_device_raw_strings": int(meta["device"].nunique()),
        "n_device_normalised_strings": int(meta["device_norm"].nunique()),
        "n_visits_device_unknown": int((meta["device_recorded"] & ~meta["device_known"]).sum()),
    }
