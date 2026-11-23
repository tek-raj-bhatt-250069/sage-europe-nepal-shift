"""Thesis figures from the aggregate tables and saved predictions (PNG and PDF in outputs/figures/).

    f1_confusion         confusion matrices on raw frames and fov crops: both paper-recipe re-runs
                         (seed 0) and the DINOv2 probe
    f2_class_gaps        per-class F1 gap (ceiling minus transfer) with 95% CIs, by meaning group
    f3_label_efficiency  macro-F1 against the number of Nepali training patients (RQ3)
    f4_recipe_inputs     paper-recipe re-runs scored on raw, masked and fov-cropped frames

    python scripts/11_figures.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sage_shift.labelspace import MEANING_GROUP, SIX_CLASSES  # noqa: E402
from sage_shift.outdir import check_out_dir  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3 (validated all-pairs)
BLUES = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, INK_2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BAND = "#efeee9"  # target zone in F3, one step off the surface
SHORT = {"esophagitis": "Esophagitis", "fundus-retroflexion": "Fundus (retrofl.)", "polyp": "Polyp",
         "pylorus-normal": "Pylorus", "rectum-retroflexion": "Rectum (retrofl.)", "z-line-normal": "Z-line"}
BACKBONE_NAMES = {"resnet50_imagenet": "ResNet-50 ImageNet", "resnet50_dino_imagenet": "ResNet-50 DINO",
                  "resnet50_gastronet_dino": "ResNet-50 GastroNet", "dinov2_vitb14": "DINOv2 ViT-B/14"}


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


def confusion_from(path: Path, test: pd.DataFrame) -> np.ndarray:
    proba = pd.read_csv(path, dtype={"sample_uuid": str}).set_index("sample_uuid").loc[test["sample_uuid"], list(SIX_CLASSES)]
    y = test["class"].map({c: i for i, c in enumerate(SIX_CLASSES)}).to_numpy()
    cm = np.zeros((len(SIX_CLASSES), len(SIX_CLASSES)), dtype=int)
    np.add.at(cm, (y, proba.to_numpy().argmax(axis=1)), 1)
    return cm


def f1_confusion(outputs: Path, test: pd.DataFrame, out_dir: Path) -> None:
    rows = [("Paper recipe, faithful (seed 0)", outputs / "paper_recipe" / "faithful_seed0_sage_probs.csv",
             outputs / "paper_recipe" / "faithful_seed0_sage_fov_probs.csv"),
            ("Paper recipe, adjusted (seed 0)", outputs / "paper_recipe" / "adjusted_seed0_sage_probs.csv",
             outputs / "paper_recipe" / "adjusted_seed0_sage_fov_probs.csv"),
            ("DINOv2 linear probe", outputs / "rq1" / "predictions" / "dinov2_vitb14__raw__transfer.csv",
             outputs / "rq1" / "predictions" / "dinov2_vitb14__fov__transfer.csv")]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blues", BLUES)
    fig, axes = plt.subplots(3, 2, figsize=(6.6, 9.4))
    labels = [SHORT[c] for c in SIX_CLASSES]
    for r, (title, raw, fov) in enumerate(rows):
        for c, (path, column) in enumerate(((raw, "raw frames"), (fov, "fov crops"))):
            ax, cm = axes[r, c], confusion_from(path, test)
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
            ax.set_title(f"{title}\n{column}: macro-F1 {macro_f1(cm):.2f}", loc="left", fontsize=8.5)
            if c == 0:
                ax.set_ylabel("True class")
            if r == 2:
                ax.set_xlabel("Predicted class")
    fig.suptitle("Where the 158 SAGE test images go (cell shade: share of the true class)", x=0.02, ha="left",
                 fontsize=9.5, color=INK, y=1.0)
    fig.tight_layout()
    save(fig, out_dir, "f1_confusion")


def macro_f1(cm: np.ndarray) -> float:
    tp, support, predicted = np.diag(cm), cm.sum(axis=1), cm.sum(axis=0)
    denom = support + predicted
    f1 = np.divide(2 * tp, denom, out=np.zeros(len(tp)), where=denom > 0)
    return float(f1[support > 0].mean())


def f2_class_gaps(table: pd.DataFrame, out_dir: Path) -> None:
    gaps = table[(table["section"] == "meaning_groups") & (table["row"] == "raw") & table["metric"].str.startswith("f1_gap_")]
    order = [c for c in SIX_CLASSES if MEANING_GROUP[c] == "shifted"] + [c for c in SIX_CLASSES if MEANING_GROUP[c] == "matched"]
    backbones = [b for b in BACKBONE_NAMES if b in set(gaps["model"])]
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
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
    ax.axhline(1.5, color=GRID, linewidth=0.8)
    ax.set_yticks(range(len(order)), [SHORT[c] for c in order])
    ax.invert_yaxis()
    ax.text(1.0, 0.5, "meaning-shifted", transform=ax.get_yaxis_transform(), ha="right", va="center", color=MUTED, fontsize=7.5)
    ax.text(1.0, 3.5, "meaning-matched", transform=ax.get_yaxis_transform(), ha="right", va="center", color=MUTED, fontsize=7.5)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.set_xlabel("F1 gap: SAGE-trained ceiling minus HyperKvasir-trained probe (raw frames), 95% CI")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.14), ncol=len(backbones), fontsize=7.5, handletextpad=0.3,
              borderaxespad=0)
    ax.set_title("Per-class transfer gap by meaning group", loc="left")
    fig.tight_layout()
    save(fig, out_dir, "f2_class_gaps")


def f3_label_efficiency(curves: pd.DataFrame, summary_rows: dict, out_dir: Path) -> None:
    variants = [v for v in ("raw", "fov_colour") if v in set(curves["variant"])]
    backbones = [b for b in BACKBONE_NAMES if b in set(curves["backbone"])]
    ks = sorted(curves["k"].unique())
    position = {k: i for i, k in enumerate(ks)}
    fig, axes = plt.subplots(len(variants), len(backbones), figsize=(7.4, 2.6 * len(variants) + 0.4), sharey=True,
                             squeeze=False)
    regimes = [("hk_plus_k", "HyperKvasir + k Nepali patients", SERIES[0]), ("sage_only", "Nepali patients only", SERIES[1])]
    for r, variant in enumerate(variants):
        for c, backbone in enumerate(backbones):
            ax = axes[r, c]
            entry = summary_rows.get(f"{backbone}/{variant}", {})
            if entry:
                ax.axhspan(entry["target_macro_f1"], entry["ceiling_macro_f1"], color=BAND, linewidth=0, zorder=0)
                ax.axhline(entry["ceiling_macro_f1"], color=MUTED, linewidth=0.9, zorder=1)
            for regime, label, colour in regimes:
                part = curves[(curves["backbone"] == backbone) & (curves["variant"] == variant) & (curves["regime"] == regime)].sort_values("k")
                x = part["k"].map(position).to_numpy()
                ax.fill_between(x, part["ci_low"], part["ci_high"], color=colour, alpha=0.12, linewidth=0)
                ax.plot(x, part["macro_f1_mean"], "-o", color=colour, linewidth=1.5, markersize=4,
                        markeredgecolor="white", markeredgewidth=0.8, label=label)
            ax.set_xticks(range(len(ks)), [str(k) for k in ks])
            ax.set_ylim(0, 1)
            ax.grid(axis="y")
            ax.set_axisbelow(True)
            if r == 0:
                ax.set_title(BACKBONE_NAMES[backbone], loc="left")
            if c == 0:
                ax.set_ylabel(f"Macro-F1 ({variant.replace('_', ' + ')})")
            if r == len(variants) - 1:
                ax.set_xlabel("Nepali training patients (k)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles += [matplotlib.lines.Line2D([], [], color=MUTED, linewidth=0.9), matplotlib.patches.Patch(color=BAND)]
    labels += ["SAGE-only ceiling (all 145 patients)", "90–100% of the ceiling (target)"]
    fig.legend(handles, labels, loc="upper left", ncol=2, fontsize=7.5, bbox_to_anchor=(0.01, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, out_dir, "f3_label_efficiency")


def f4_recipe_inputs(table: pd.DataFrame, out_dir: Path) -> None:
    runs = table[(table["section"] == "paper_recipe") & table["model"].str.match(r"(faithful|adjusted)_seed\d+$")
                 & (table["metric"] == "macro_f1") & table["row"].isin(["raw", "masked", "fov"])]
    inputs = ["raw", "masked", "fov"]
    names = {"raw": "raw frames", "masked": "border masked", "fov": "fov crop"}
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    constant = table[(table["section"] == "paper_recipe") & (table["model"] == "constant") & (table["metric"] == "macro_f1")]["estimate"].max()
    ax.axhline(constant, color=MUTED, linewidth=0.8)
    ax.text(2.25, constant + 0.01, "best constant\npredictor", fontsize=7, color=MUTED, va="bottom", ha="right")
    for recipe, colour, shift in (("faithful", SERIES[0], -0.08), ("adjusted", SERIES[1], 0.08)):
        first = True
        for model, part in runs[runs["model"].str.startswith(recipe)].groupby("model"):
            part = part.set_index("row").reindex(inputs)
            x = np.arange(len(inputs)) + shift
            ok = part["estimate"].notna().to_numpy()
            for i in range(len(inputs) - 1):  # join neighbouring inputs only: no line over a missing score
                if ok[i] and ok[i + 1]:
                    ax.plot(x[i:i + 2], part["estimate"].to_numpy()[i:i + 2], "-", color=colour, linewidth=1.0, alpha=0.5)
            ax.vlines(x[ok], part["ci_low"].to_numpy()[ok], part["ci_high"].to_numpy()[ok], color=colour, linewidth=1.0, alpha=0.6)
            ax.plot(x[ok], part["estimate"].to_numpy()[ok], "o", color=colour, markersize=5, markeredgecolor="white",
                    markeredgewidth=1.0, label=f"{recipe} recipe" if first else None)
            first = False
    ax.set_xticks(range(len(inputs)), [names[i] for i in inputs])
    ax.set_xlim(-0.4, 2.4)
    ax.set_ylim(0, 0.8)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.set_ylabel("Macro-F1 on SAGE test (95% CI)")
    ax.set_title("The same fine-tuned models, three ways of presenting the SAGE frames", loc="left")
    ax.legend(loc="upper left", fontsize=7.5)
    fig.tight_layout()
    save(fig, out_dir, "f4_recipe_inputs")


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
    f2_class_gaps(table, args.out_dir)
    f4_recipe_inputs(table, args.out_dir)
    curves_path = args.outputs / "rq3" / "rq3_curves.csv"
    if curves_path.is_file():
        summary = json.loads((args.outputs / "rq3" / "rq3_summary.json").read_text(encoding="utf-8"))["results"]
        f3_label_efficiency(pd.read_csv(curves_path), summary, args.out_dir)
    print(f"Wrote figures to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
