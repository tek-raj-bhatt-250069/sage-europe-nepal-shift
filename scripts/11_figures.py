"""Thesis figures from the aggregate tables and saved predictions (PNG and PDF in outputs/figures/).

    f1_confusion         confusion matrices on raw frames and field crops: both paper-recipe re-runs
                         (seed 0) and the DINOv2 probe
    f2_class_gaps        per-class F1 gap (SAGE-only reference minus transfer) with 95% CIs, by meaning
                         group, with the test images and patients behind each class
    f3_label_efficiency  macro-F1 against the number of Nepali training patients (RQ3), plus macro-AUROC
                         rows when outputs/exploratory/rq3_auroc_curves.csv exists
    f4_recipe_inputs     paper-recipe re-runs scored on raw, masked and field-cropped SAGE frames
    f5_causal_diagram    conceptual diagram of the site factors and where each intervention acts (no data)
    f6_loo_influence     leave-one-patient-out influence on the headline gaps, when
                         outputs/sensitivity/loo_influence.csv exists
    f7_geometry_controls re-trained fine-tuned models under the geometry inputs: top-class share on SAGE
                         and HyperKvasir frames placed into SAGE's layout, when
                         outputs/exploratory_gpu/geometry_table.csv exists

Figures never show dataset images, IDs or patient codes. Optional inputs that are missing are skipped.

    python scripts/11_figures.py
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from sage_shift.labelspace import MEANING_GROUP, SIX_CLASSES  # noqa: E402
from sage_shift.outdir import check_out_dir  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3 (validated all-pairs)
BLUES = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, INK_2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BAND = "#efeee9"  # target zone in F3 and diagram boxes, one step off the surface
SHORT = {"esophagitis": "Esophagitis", "fundus-retroflexion": "Fundus (retrofl.)", "polyp": "Polyp",
         "pylorus-normal": "Pylorus", "rectum-retroflexion": "Rectum (retrofl.)", "z-line-normal": "Z-line"}
BACKBONE_NAMES = {"resnet50_imagenet": "ResNet-50 ImageNet", "resnet50_dino_imagenet": "ResNet-50 DINO",
                  "resnet50_gastronet_dino": "ResNet-50 GastroNet", "dinov2_vitb14": "DINOv2 ViT-B/14"}
REFERENCE = "SAGE-only reference"  # the SAGE-trained probe (files and columns call it the ceiling)
FOV_TEST_TIME = "Exploratory: test-time field crop of SAGE"
INPUTS = {"raw": "raw frame", "masked": "masked\n(border blacked out,\nnot cropped)", "fov": "test-time\nfield crop"}
AUROC_COLUMNS = ["backbone", "variant", "regime", "k", "auroc_mean", "ci_low", "ci_high", "draws"]
LOO_COLUMNS = ["backbone_or_run", "variant", "quantity", "patient_group", "n_images", "delta"]
HEADLINE_GAPS = {"gap_macro_f1": f"Transfer gap: {REFERENCE} − transfer (macro-F1)",
                 "macro_f1_difference": "Exploratory: test-time field crop − raw frames (macro-F1)"}
PROBE_VARIANTS = ("raw", "fov", "colour", "fov_colour")
GEOMETRY_COLUMNS = ["section", "model", "row", "metric", "estimate", "ci_low", "ci_high"]
SAGE_GEOMETRY = {"raw": "raw\nframe", "masked": "masked\n(black)", "masked_grey": "masked\n(grey)",
                 "letterbox": "letterbox\n(black)", "letterbox_grey": "letterbox\n(grey)",
                 "fov": "field crop\n(black)", "fov_grey": "field crop\n(grey)"}
HK_GEOMETRY = {"raw": "raw\nframe", "sageify": "SAGE\nlayout\n(black)", "sageify_grey": "SAGE\nlayout\n(grey)"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Thesis figures.")
    parser.add_argument("--outputs", type=Path, default=ROOT / "outputs")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "figures")
    return parser.parse_args()


def style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "grid.color": GRID, "grid.linewidth": 0.6,
        "grid.linestyle": "-", "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "savefig.dpi": 220, "savefig.bbox": "tight", "figure.facecolor": "white", "axes.facecolor": "white"})


def save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{suffix}")
    plt.close(fig)


def note(fig: plt.Figure, text: str, y: float) -> None:
    """A small left-aligned note in secondary ink (figure coordinates)."""
    fig.text(0.01, y, text, ha="left", va="top", fontsize=7.5, color=INK_2, linespacing=1.4)


def optional_table(path: Path, columns: list[str]) -> pd.DataFrame | None:
    """The table at `path` when it exists and has `columns`; otherwise None, with a message."""
    if not path.is_file():
        print(f"Skipped (not found yet): {path}")
        return None
    try:
        frame = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as error:
        print(f"Skipped (unreadable): {path}: {error}")
        return None
    missing = [column for column in columns if column not in frame.columns]
    if missing or frame.empty:
        print(f"Skipped (missing columns {missing} or no rows): {path}")
        return None
    return frame


def pretty(name: str) -> str:
    """Readable label for a backbone, run or quantity name."""
    if name in BACKBONE_NAMES:
        return BACKBONE_NAMES[name]
    run = re.fullmatch(r"(faithful|adjusted)_seed(\d+)", name)
    if run:
        return f"{run[1].capitalize()} recipe, seed {run[2]}"
    text = name.replace("ceiling", REFERENCE).replace("macro_f1", "macro-F1").replace("_", " ")
    return text[:1].upper() + text[1:]


def confusion_from(path: Path, test: pd.DataFrame) -> np.ndarray:
    proba = pd.read_csv(path, dtype={"sample_uuid": str}).set_index("sample_uuid").loc[test["sample_uuid"], list(SIX_CLASSES)]
    y = test["class"].map({c: i for i, c in enumerate(SIX_CLASSES)}).to_numpy()
    cm = np.zeros((len(SIX_CLASSES), len(SIX_CLASSES)), dtype=int)
    np.add.at(cm, (y, proba.to_numpy().argmax(axis=1)), 1)
    return cm


def f1_confusion(outputs: Path, test: pd.DataFrame, out_dir: Path) -> None:
    recipe, probes = outputs / "paper_recipe", outputs / "rq1" / "predictions"
    rows = [("Paper recipe, faithful (seed 0)", recipe / "faithful_seed0_sage_probs.csv",
             recipe / "faithful_seed0_sage_fov_probs.csv", FOV_TEST_TIME),
            ("Paper recipe, adjusted (seed 0)", recipe / "adjusted_seed0_sage_probs.csv",
             recipe / "adjusted_seed0_sage_fov_probs.csv", FOV_TEST_TIME),
            ("DINOv2 linear probe", probes / "dinov2_vitb14__raw__transfer.csv",
             probes / "dinov2_vitb14__fov__transfer.csv", "Field crop of both datasets")]
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    labels = [SHORT[c] for c in SIX_CLASSES]
    fig = plt.figure(figsize=(6.6, 9.5), layout="constrained")  # fits an A4 text block at text width
    fig.suptitle(f"Where the {len(test)} SAGE test images go ({test['patient_group'].nunique()} patients; "
                 "cell shade: share of the true class)", x=0.01, ha="left", fontsize=9.5, color=INK)
    for r, (subfig, (title, raw, fov, fov_name)) in enumerate(zip(fig.subfigures(len(rows), 1), rows)):
        subfig.suptitle(title, x=0.01, ha="left", fontsize=9, color=INK, fontweight="bold")
        for c, (ax, path, column) in enumerate(zip(subfig.subplots(1, 2), (raw, fov), ("Raw frames", fov_name))):
            cm = confusion_from(path, test)
            share = cm / cm.sum(axis=1, keepdims=True)
            ax.imshow(share, cmap=cmap, vmin=0, vmax=1)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    if cm[i, j]:
                        ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7.5,
                                color="white" if share[i, j] > 0.55 else INK)
            ax.set_xticks(range(6), labels, rotation=40, ha="right")
            ax.set_yticks(range(6), labels if c == 0 else [""] * 6)
            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_title(f"{column}\nmacro-F1 {macro_f1(cm):.2f}", loc="left", fontsize=8.5)
            if c == 0:
                ax.set_ylabel("True class")
            if r == len(rows) - 1:
                ax.set_xlabel("Predicted class")
    fig.supxlabel("Paper-recipe models: training seed 0 is shown; seeds 1–2 are reported in the per-seed table. "
                  "These models were\ntrained on raw HyperKvasir frames, and only SAGE was cropped, at test time "
                  "(exploratory). The DINOv2 probe's\nfield-crop variant crops both datasets.",
                  x=0.01, ha="left", fontsize=7.5, color=INK_2, linespacing=1.4)
    save(fig, out_dir, "f1_confusion")


def macro_f1(cm: np.ndarray) -> float:
    tp, support, predicted = np.diag(cm), cm.sum(axis=1), cm.sum(axis=0)
    denom = support + predicted
    f1 = np.divide(2 * tp, denom, out=np.zeros(len(tp)), where=denom > 0)
    return float(f1[support > 0].mean())


def f2_class_gaps(table: pd.DataFrame, test: pd.DataFrame, out_dir: Path) -> None:
    gaps = table[(table["section"] == "meaning_groups") & (table["row"] == "raw") & table["metric"].str.startswith("f1_gap_")]
    order = [c for c in SIX_CLASSES if MEANING_GROUP[c] == "shifted"] + [c for c in SIX_CLASSES if MEANING_GROUP[c] == "matched"]
    n_images, n_patients = test["class"].value_counts(), test.groupby("class")["patient_group"].nunique()
    backbones = [b for b in BACKBONE_NAMES if b in set(gaps["model"])]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    offsets = np.linspace(-0.22, 0.22, len(backbones))
    for b, (backbone, offset) in enumerate(zip(backbones, offsets)):
        part = gaps[gaps["model"] == backbone].set_index("metric")
        ys = np.arange(len(order)) + offset
        est = [part.loc[f"f1_gap_{c}", "estimate"] for c in order]
        low = [part.loc[f"f1_gap_{c}", "ci_low"] for c in order]
        high = [part.loc[f"f1_gap_{c}", "ci_high"] for c in order]
        ax.hlines(ys, low, high, color=SERIES[b], linewidth=1.4)
        ax.plot(est, ys, "o", color=SERIES[b], markersize=5, markeredgecolor="white", markeredgewidth=1.0,
                label=BACKBONE_NAMES[backbone])
    ax.axvline(0, color=AXIS, linewidth=0.8)
    n_shifted = sum(MEANING_GROUP[c] == "shifted" for c in order)
    ax.axhline(n_shifted - 0.5, color=GRID, linewidth=0.8)
    ax.set_yticks(range(len(order)), [f"{SHORT[c]}\n({n_images[c]} img / {n_patients[c]} pt)" for c in order])
    ax.invert_yaxis()
    for text, y in (("meaning-shifted", (n_shifted - 1) / 2), ("meaning-matched", (n_shifted + len(order) - 1) / 2)):
        ax.text(1.0, y, text, transform=ax.get_yaxis_transform(), ha="right", va="center", color=MUTED, fontsize=7.5)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.set_xlabel(f"{REFERENCE} − transfer (F1)")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=len(backbones), fontsize=7.5, handletextpad=0.3,
              borderaxespad=0)
    fig.suptitle("Descriptive: per-class transfer gap on raw frames, by meaning group", x=0.01, ha="left",
                 fontsize=9.5, color=INK)
    fig.tight_layout(rect=(0, 0.12, 1, 0.98))
    note(fig, f"Points: F1 of the {REFERENCE} (trained on SAGE) minus F1 of the probe trained on HyperKvasir;\n"
              "lines: 95% CIs from the patient-level bootstrap. Meaning groups as pre-registered.\n"
              "img / pt: SAGE test images / patients of the class.", 0.1)
    save(fig, out_dir, "f2_class_gaps")


def _curve_rows(curves: pd.DataFrame, auroc: pd.DataFrame | None, variants: list[str]) -> list[tuple[str, pd.DataFrame, str, str]]:
    """(metric label, table, mean column, variant) for every row of panels in F3."""
    rows = [("Macro-F1", curves, "macro_f1_mean", v) for v in variants]
    if auroc is not None:
        rows += [("Macro-AUROC", auroc, "auroc_mean", v) for v in variants if v in set(auroc["variant"])]
    return rows


def f3_label_efficiency(curves: pd.DataFrame, summary_rows: dict, out_dir: Path, auroc: pd.DataFrame | None = None) -> None:
    variants = [v for v in ("raw", "fov_colour") if v in set(curves["variant"])]
    backbones = [b for b in BACKBONE_NAMES if b in set(curves["backbone"])]
    ks = sorted(set(curves["k"]) | (set(auroc["k"]) if auroc is not None else set()))
    position = {k: i for i, k in enumerate(ks)}
    rows = _curve_rows(curves, auroc, variants)
    height = (2.3 if len(rows) <= 2 else 2.0) * len(rows) + 1.3  # four rows still fit one A4 page
    fig, axes = plt.subplots(len(rows), len(backbones), figsize=(7.4, height), squeeze=False)
    regimes = [("hk_plus_k", "HyperKvasir + k Nepali patients", SERIES[0]), ("sage_only", "Nepali patients only", SERIES[1])]
    for r, (metric, frame, column, variant) in enumerate(rows):
        low_limit = 0.0 if metric == "Macro-F1" else min(0.5, np.floor(frame["ci_low"].min() * 10) / 10)
        for c, backbone in enumerate(backbones):
            ax = axes[r, c]
            panel = frame[(frame["backbone"] == backbone) & (frame["variant"] == variant)]
            if metric == "Macro-F1":
                entry = summary_rows.get(f"{backbone}/{variant}", {})
                if entry:
                    ax.axhspan(entry["target_macro_f1"], entry["ceiling_macro_f1"], color=BAND, linewidth=0, zorder=0)
                    ax.axhline(entry["ceiling_macro_f1"], color=MUTED, linewidth=0.9, zorder=1)
            else:
                reference = panel[(panel["regime"] == "sage_only") & (panel["k"] == panel["k"].max())]
                if len(reference):
                    ax.axhline(reference[column].iloc[0], color=MUTED, linewidth=0.9, zorder=1)
            for regime, label, colour in regimes:
                part = panel[panel["regime"] == regime].sort_values("k")
                x = part["k"].map(position).to_numpy()
                ax.fill_between(x, part["ci_low"], part["ci_high"], color=colour, alpha=0.12, linewidth=0)
                ax.plot(x, part[column], "-o", color=colour, linewidth=1.5, markersize=4,
                        markeredgecolor="white", markeredgewidth=0.8, label=label)
            ax.set_xticks(range(len(ks)), [str(k) for k in ks])
            ax.set_ylim(low_limit, 1)
            ax.tick_params(labelleft=c == 0)
            ax.grid(axis="y")
            ax.set_axisbelow(True)
            if r == 0:
                ax.set_title(BACKBONE_NAMES[backbone], loc="left")
            if c == 0:
                ax.set_ylabel(f"{metric} ({variant.replace('_', ' + ')})")
            if r == len(rows) - 1 and c == len(backbones) // 2:  # one centred label for the shared k axis
                ax.set_xlabel("Nepali training patients, k (categorical axis, not to scale)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles += [Line2D([], [], color=MUTED, linewidth=0.9), Patch(color=BAND)]
    labels += [f"{REFERENCE} (all {max(ks)} patients)",
               f"90–100% of the {REFERENCE} (target{', macro-F1' if auroc is not None else ''})"]
    fig.legend(handles, labels, loc="upper left", ncol=2, fontsize=7.5, bbox_to_anchor=(0.01, 1.0))
    fig.tight_layout(rect=(0, 0.78 / height, 1, 1 - 0.55 / height))
    single = sorted(curves.loc[curves["draws"] == 1, "k"].unique())
    note(fig, f"Lines: mean over {int(curves['draws'].max())} random draws of k patients (a single fit at k = "
              f"{' and '.join(str(k) for k in single)}). Bands: 95% intervals of that mean from the\ntest-patient "
              "bootstrap, not the draw-to-draw spread. The k axis is categorical (numbers of Nepali patients, "
              "not to scale).", 0.5 / height)
    save(fig, out_dir, "f3_label_efficiency")


def f4_recipe_inputs(table: pd.DataFrame, out_dir: Path) -> None:
    rows = table[(table["section"] == "paper_recipe") & table["model"].str.match(r"(faithful|adjusted)_seed\d+$")]
    runs = rows[(rows["metric"] == "macro_f1") & rows["row"].isin(list(INPUTS))]
    epochs = rows.groupby("model")["epochs"].max()
    inputs = list(INPUTS)
    fig, ax = plt.subplots(figsize=(6.0, 4.3))
    constant = table[(table["section"] == "paper_recipe") & (table["model"] == "constant") & (table["metric"] == "macro_f1")]["estimate"].max()
    ax.axhline(constant, color=MUTED, linewidth=0.8)
    ax.text(2.45, constant + 0.01, "best constant\npredictor", fontsize=7, color=MUTED, va="bottom", ha="right")
    for recipe, colour, shift in (("faithful", SERIES[0], -0.17), ("adjusted", SERIES[1], 0.17)):
        for model, part in runs[runs["model"].str.startswith(recipe)].groupby("model"):
            seed = int(model.rsplit("seed", 1)[1])
            marker = "s" if seed == 0 else "o"
            part = part.set_index("row").reindex(inputs)
            x = np.arange(len(inputs)) + shift + 0.07 * (seed - 1)  # dodge the seeds so every CI stays visible
            ok = part["estimate"].notna().to_numpy()
            for i in range(len(inputs) - 1):  # join neighbouring inputs only: no line over a missing score
                if ok[i] and ok[i + 1]:
                    ax.plot(x[i:i + 2], part["estimate"].to_numpy()[i:i + 2], "-", color=colour, linewidth=1.0, alpha=0.45)
            ax.vlines(x[ok], part["ci_low"].to_numpy()[ok], part["ci_high"].to_numpy()[ok], color=colour, linewidth=1.0, alpha=0.6)
            ax.plot(x[ok], part["estimate"].to_numpy()[ok], marker, color=colour, markersize=5.5 if seed == 0 else 5,
                    markeredgecolor="white", markeredgewidth=1.0, linestyle="none")
    ax.set_xticks(range(len(inputs)), [INPUTS[i] for i in inputs])
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(0, 0.8)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.set_ylabel("Macro-F1 on SAGE test (95% CI)")
    later = sorted({int(e) for m, e in epochs.items() if m.startswith("faithful") and not m.endswith("seed0")})
    handles = [Line2D([], [], color=SERIES[0], marker="o", markeredgecolor="white", linewidth=1.0),
               Line2D([], [], color=SERIES[1], marker="o", markeredgecolor="white", linewidth=1.0),
               Line2D([], [], color=INK_2, marker="s", markeredgecolor="white", linestyle="none"),
               Line2D([], [], color=INK_2, marker="o", markeredgecolor="white", linestyle="none")]
    labels = ["faithful recipe", "adjusted recipe", f"seed 0 (faithful: {int(epochs['faithful_seed0'])} epochs)",
              f"seeds 1–2 (faithful: {', '.join(str(e) for e in later)} epochs)"]
    ax.legend(handles, labels, loc="upper left", fontsize=7.5, ncol=2, columnspacing=1.2)
    fig.suptitle("Exploratory: the same fine-tuned models, three ways of presenting the SAGE frames", x=0.01,
                 ha="left", fontsize=9.5, color=INK)
    fig.tight_layout(rect=(0, 0.11, 1, 0.98))
    adjusted = sorted({int(e) for m, e in epochs.items() if m.startswith("adjusted")})
    note(fig, "Both recipes were trained on raw HyperKvasir frames; SAGE frames were masked or cropped at test time "
              f"only.\nMasked frames were scored for seeds 1–2 only. The adjusted recipe ran "
              f"{', '.join(str(e) for e in adjusted)} epochs for every seed. Seeds are\nshown side by side "
              "(0, 1, 2 from left to right within each recipe).", 0.085)
    save(fig, out_dir, "f4_recipe_inputs")


def _box(ax: plt.Axes, cx: float, cy: float, w: float, h: float, title: str, sub: str = "",
         fill: str = BAND, edge: str = AXIS, size: float = 8.5) -> dict[str, float]:
    """A rounded box with a title and an optional second line; returns its edges."""
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.05",
                                facecolor=fill, edgecolor=edge, linewidth=0.9))
    ax.text(cx, cy + (0.09 if sub else 0), title, ha="center", va="center", fontsize=size, color=INK,
            fontweight="bold", linespacing=1.3)
    if sub:
        ax.text(cx, cy - 0.11, sub, ha="center", va="center", fontsize=7.5, color=INK_2)
    return {"left": cx - w / 2, "right": cx + w / 2, "top": cy + h / 2, "bottom": cy - h / 2, "cx": cx, "cy": cy}


def _arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], colour: str = INK_2,
           width: float = 0.9) -> None:
    ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "-|>", "color": colour, "linewidth": width,
                                                      "shrinkA": 0, "shrinkB": 0, "mutation_scale": 9})


def f5_causal_diagram(out_dir: Path) -> None:
    """Conceptual diagram (no data): site factors, the format-class link in HyperKvasir and the interventions."""
    width, height = 6.5, 4.75
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")
    ax.text(0.08, height - 0.1, "What differs between the two sites, and where each intervention acts (conceptual)",
            ha="left", va="top", fontsize=9.5, color=INK)
    factor_x, factor_w = 3.03, 2.5
    factors = [("Processor and frame format", "field layout, borders, overlays, colour", 3.3),
               ("Population and disease spectrum", "which findings occur, how they look", 2.58),
               ("Capture habits and frame selection", "views, framing, which stills are kept", 1.86),
               ("Annotation conventions", "class definitions and boundaries", 1.14)]
    boxes = [_box(ax, factor_x, y, factor_w, 0.54, title, sub, fill="white") for title, sub, y in factors]
    site = _box(ax, 0.73, 2.22, 1.3, 2.7, "")
    ax.text(site["cx"], 3.25, "Site", ha="center", va="center", fontsize=9, color=INK, fontweight="bold")
    ax.text(site["cx"], 2.1, "one Norwegian\nhospital\n(HyperKvasir)\n\nvs\n\none Nepali\nhospital\n(SAGE)",
            ha="center", va="center", fontsize=8, color=INK, linespacing=1.3)
    for box in boxes:
        _arrow(ax, (site["right"], box["cy"]), (box["left"], box["cy"]))
    image = _box(ax, 5.53, 2.95, 0.9, 0.55, "Image X")
    label = _box(ax, 5.53, 1.55, 0.9, 0.55, "Label Y")
    processor, population, capture, annotation = boxes
    edges = [(processor, image, 0.14), (population, image, 0.0), (capture, image, -0.14),
             (population, label, 0.14), (capture, label, 0.0), (annotation, label, -0.14)]
    for source, target, dy in edges:
        _arrow(ax, (source["right"], source["cy"] - (0.08 if target is label and source is population else 0)),
               (target["left"], target["cy"] + dy))

    # interventions, each attached to the factor it acts on
    blue_fill = BLUES[1]
    for cx, w, name, target in ((1.98, 1.86, "Field-of-view crop", "acts on frame format / geometry"),
                                (3.78, 1.46, "Colour normalisation", "acts on colour")):
        box = _box(ax, cx, 4.08, w, 0.44, name, target, fill=blue_fill, edge=SERIES[0], size=8)
        _arrow(ax, (cx, box["bottom"]), (cx, processor["top"]), colour=SERIES[0], width=1.2)
    prior = _box(ax, label["cx"], 0.8, 1.42, 0.44, "Prior correction (EM)", "acts on class priors",
                 fill=blue_fill, edge=SERIES[0], size=8)
    _arrow(ax, (label["cx"], prior["top"]), (label["cx"], label["bottom"]), colour=SERIES[0], width=1.2)

    # HyperKvasir only: frame format goes with class (dashed, both directions)
    y_link, x_route = processor["cy"] + 0.17, 6.28
    ax.plot([processor["right"] + 0.09, x_route, x_route, label["right"] + 0.09],
            [y_link, y_link, label["cy"], label["cy"]], color=SERIES[1], linewidth=1.2, linestyle=(0, (4, 2.5)))
    _arrow(ax, (processor["right"] + 0.1, y_link), (processor["right"], y_link), colour=SERIES[1], width=1.2)
    _arrow(ax, (label["right"] + 0.1, label["cy"]), (label["right"], label["cy"]), colour=SERIES[1], width=1.2)
    ax.text(x_route, y_link + 0.07, "HyperKvasir only: frame\nformat goes with class\n(lower-GI classes mostly\n"
            "in the older SD format)", ha="right", va="bottom", fontsize=7.5, color=INK_2, linespacing=1.3)

    # key and the identifiability note
    y_key = 0.36
    _arrow(ax, (0.1, y_key), (0.45, y_key))
    ax.text(0.52, y_key, "causal influence", ha="left", va="center", fontsize=7.5, color=INK_2)
    ax.plot([1.6, 1.95], [y_key, y_key], color=SERIES[1], linewidth=1.2, linestyle=(0, (4, 2.5)))
    ax.text(2.02, y_key, "association in HyperKvasir only", ha="left", va="center", fontsize=7.5, color=INK_2)
    ax.add_patch(FancyBboxPatch((3.85, y_key - 0.07), 0.3, 0.14, boxstyle="round,pad=0,rounding_size=0.03",
                                facecolor=blue_fill, edgecolor=SERIES[0], linewidth=0.9))
    ax.text(4.22, y_key, "intervention tested (RQ2)", ha="left", va="center", fontsize=7.5, color=INK_2)
    ax.text(0.08, 0.1, "With one hospital per side, population differences cannot be separated from the other "
            "site factors.", ha="left", va="center", fontsize=8, color=INK)
    save(fig, out_dir, "f5_causal_diagram")


def _label_points(ax: plt.Axes, points: list[tuple[float, float, str]], size: float = 7) -> None:
    """Label points above or below them, moving a label further out when it would overlap an earlier one."""
    scale = ax.figure.dpi / 72  # display pixels per point
    placed: list[tuple[float, float, float]] = []
    for x, y, text in points:
        px, py = ax.transData.transform((x, y))
        half = 0.32 * size * len(text) * scale  # rough half width of the label
        for dy in (8, -8, 17, -17, 26):
            centre = py - dy * scale  # the y axis is inverted, so a positive offset is drawn above
            if all(abs(centre - cy) > size * 1.15 * scale or abs(px - cx) > half + ch + 2 * scale
                   for cx, cy, ch in placed):
                break
        placed.append((px, centre, half))
        ax.annotate(text, (x, y), xytext=(0, dy), textcoords="offset points", ha="center", va="center",
                    fontsize=size, color=INK_2)


def f6_loo_influence(loo: pd.DataFrame, out_dir: Path) -> None:
    """Leave-one-patient-out deltas per headline gap; the largest are labelled by the patient's image count."""
    headline = loo[loo["quantity"].isin(list(HEADLINE_GAPS))]
    loo = headline if len(headline) else loo  # unfamiliar quantity names: show them all
    quantities = list(dict.fromkeys(loo["quantity"]))
    strips = {q: list(loo[loo["quantity"] == q].groupby(["backbone_or_run", "variant"], sort=False)) for q in quantities}
    n_strips = [len(strips[q]) for q in quantities]
    height = 0.5 * sum(n_strips) + 0.8 * len(quantities) + 1.0
    fig, axes = plt.subplots(len(quantities), 1, figsize=(6.4, height), squeeze=False,
                             gridspec_kw={"height_ratios": n_strips})
    rng = np.random.default_rng(0)  # vertical jitter only
    labels: list[tuple[plt.Axes, list[tuple[float, float, str]]]] = []
    for ax, quantity in zip(axes[:, 0], quantities):
        names, points = [], []
        for i, ((name, variant), strip) in enumerate(strips[quantity]):
            shown = f" · {variant.replace('_', ' + ')}" if variant in PROBE_VARIANTS else ""  # runs: the title says it
            names.append(pretty(str(name)) + shown)
            order = strip["delta"].abs().sort_values(ascending=False).index
            rest = strip.loc[order[3:], "delta"].to_numpy(dtype=float)
            ax.plot(rest, i + rng.uniform(-0.15, 0.15, len(rest)), "o", color=SERIES[0], alpha=0.45,
                    markersize=3.5, markeredgewidth=0)
            for row in strip.loc[order[:3]].itertuples():
                ax.plot(row.delta, i, "o", color=SERIES[1], markersize=5.5, markeredgecolor="white", markeredgewidth=1.0)
                points.append((float(row.delta), i, f"{int(row.n_images)} img"))
        ax.axvline(0, color=AXIS, linewidth=0.8)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.set_yticks(range(len(names)), names)
        ax.set_ylim(len(names) - 0.4, -0.6)
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        ax.set_title(HEADLINE_GAPS.get(quantity, pretty(str(quantity))), loc="left")
        ax.set_xlabel("Change in the estimate when one test patient is left out")
        labels.append((ax, points))
    fig.suptitle("Leave-one-patient-out influence on the headline gaps", x=0.01, ha="left", fontsize=9.5, color=INK)
    fig.tight_layout(rect=(0, 0.62 / height, 1, 1 - 0.1 / height))
    for ax, points in labels:  # after the layout, so the overlap check sees the final positions
        _label_points(ax, points)
    note(fig, "Each dot is one test patient left out: the estimate without that patient minus the estimate with all "
              "test patients.\nThe three largest changes per row are marked in orange and labelled with that "
              "patient's number of test images.", 0.47 / height)
    save(fig, out_dir, "f6_loo_influence")


def f7_geometry_controls(table: pd.DataFrame, collapse: float, out_dir: Path) -> None:
    """Kept re-runs under the geometry inputs: SAGE top-class share and the HyperKvasir counterfactual."""
    panels = (("finetuned_sage", "top_class_share", SAGE_GEOMETRY, "Top-class share of predictions",
               "SAGE six-class test (158 images)"),
              ("finetuned_hk", "upper_gi_polyp_or_rectum_share", HK_GEOMETRY, "Share predicted polyp or rectum",
               "HyperKvasir fold 1, upper-GI frames"))
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.0), gridspec_kw={"width_ratios": [7, 3.2]})
    for ax, (section, metric, inputs, ylabel, title) in zip(axes, panels):
        rows = table[(table["section"] == section) & (table["metric"] == metric) & table["row"].isin(list(inputs))]
        tags = sorted(rows["model"].unique(), key=lambda tag: int(str(tag).rsplit("seed", 1)[1]))
        for k, tag in enumerate(tags):
            part = rows[rows["model"] == tag].set_index("row").reindex(list(inputs))
            x = np.arange(len(inputs)) + 0.16 * (k - (len(tags) - 1) / 2)  # seeds side by side
            ok = part["estimate"].notna().to_numpy()
            ax.vlines(x[ok], part["ci_low"].to_numpy()[ok], part["ci_high"].to_numpy()[ok], color=SERIES[1],
                      linewidth=1.0, alpha=0.6)
            ax.plot(x[ok], part["estimate"].to_numpy()[ok], "o", color=SERIES[1], markersize=5,
                    markeredgecolor="white", markeredgewidth=1.0, linestyle="none")
        ax.set_xticks(range(len(inputs)), list(inputs.values()), fontsize=7.5)
        ax.set_xlim(-0.5, len(inputs) - 0.5)
        ax.set_ylim(0, 1.04)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.set_ylabel(f"{ylabel} (95% CI)")
        ax.set_title(title, loc="left")
    axes[0].axhline(collapse, color=MUTED, linewidth=0.8)
    axes[0].text(len(SAGE_GEOMETRY) - 0.55, collapse - 0.02, f"collapse: top class ≥ {collapse:.0%}", fontsize=7,
                 color=MUTED, ha="right", va="top")
    fig.suptitle("Exploratory: re-trained adjusted models under geometry controls, specified before they were run",
                 x=0.01, ha="left", fontsize=9.5, color=INK)
    fig.tight_layout(rect=(0, 0.17, 1, 0.97))
    note(fig, "Each dot is one training seed (0, 1, 2 from left to right). The models were trained on raw HyperKvasir "
              "frames; inputs were changed at test time only.\nBlack or grey: the area outside the endoscopic field. "
              "Letterbox: field crop padded to a square without distortion; field crop: squashed to a square.\n"
              "SAGE layout: HyperKvasir's picture placed into SAGE's most common field box on a 768 × 576 canvas.\n"
              "Intervals: patient-level bootstrap on SAGE, image-level on HyperKvasir (no patient identifiers).", 0.12)
    save(fig, out_dir, "f7_geometry_controls")


def main() -> int:
    args = parse_args()
    check_out_dir(args.out_dir, ROOT / "data", ROOT)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    style()
    sage = pd.read_csv(args.outputs / "manifests" / "sage_manifest.csv", dtype={"class": str, "sample_uuid": str},
                       keep_default_na=False)
    test = sage[(sage["in_six"] == 1) & (sage["split"] == "test")].reset_index(drop=True)
    table = pd.read_csv(args.outputs / "rq2" / "rq2_table.csv")
    f1_confusion(args.outputs, test, args.out_dir)
    f2_class_gaps(table, test, args.out_dir)
    f4_recipe_inputs(table, args.out_dir)
    f5_causal_diagram(args.out_dir)
    curves_path = args.outputs / "rq3" / "rq3_curves.csv"
    if curves_path.is_file():
        summary = json.loads((args.outputs / "rq3" / "rq3_summary.json").read_text(encoding="utf-8"))["results"]
        auroc = optional_table(args.outputs / "exploratory" / "rq3_auroc_curves.csv", AUROC_COLUMNS)
        f3_label_efficiency(pd.read_csv(curves_path), summary, args.out_dir, auroc)
    loo = optional_table(args.outputs / "sensitivity" / "loo_influence.csv", LOO_COLUMNS)
    if loo is not None:
        f6_loo_influence(loo, args.out_dir)
    geometry = optional_table(args.outputs / "exploratory_gpu" / "geometry_table.csv", GEOMETRY_COLUMNS)
    if geometry is not None:
        config = json.loads((ROOT / "configs" / "experiment.json").read_text(encoding="utf-8"))
        collapse = float(config["paper_recipe_rerun"]["collapse_criterion"]["top_class_share_of_predictions"])
        f7_geometry_controls(geometry, collapse, args.out_dir)
    print(f"Wrote figures to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
