# Population or Pipeline? Diagnosing and Closing the Europe-to-Nepal Performance Gap in GI Endoscopy with Vision Foundation Models

Code for an MSc thesis (Data Science and Computational Intelligence, Softwarica College; degree awarded by Coventry University). This is research code only and is not a clinical or diagnostic tool.

## Data

Data not included; see `scripts/00_download_sage.py`.

The target dataset is **SAGE**, the South Asian GI Endoscopy dataset: 1,300 endoscopy images collected at Dhulikhel Hospital, Nepal.

- **Citation:** Oli et al., *Machine Learning for Biomedical Imaging (MELBA)*, 2026. doi: [10.59275/j.melba.2026-fe89](https://doi.org/10.59275/j.melba.2026-fe89)
- **Dataset code:** <https://github.com/bhattarailab/SAGE>
- **Licence and attribution:** SAGE is © 2026 Nepal Applied Mathematics and Informatics Institute for Research and is licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). This repository does not redistribute the dataset or any modified copy of it.
- **Ethics:** use of SAGE is bound by its ethical use clause (`ETHICS.md` in the dataset): research use only and no re-identification attempts.

No images, dataset files, per-patient tables or per-image tables are stored in this repository. Only code, configs and aggregate statistics are committed.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
# Optional, for feature extraction on CPU (Colab has its own PyTorch):
pip install --index-url https://download.pytorch.org/whl/cpu -r requirements-torch.txt
```

To download SAGE (needs a Synapse account with access to the dataset):

```bash
export SYNAPSE_AUTH_TOKEN=<your personal access token>
python scripts/00_download_sage.py
```

The token is read only from the environment; see `.env.example`.

## Scripts

| Script | What it does |
|---|---|
| `python scripts/00_download_sage.py` | Downloads SAGE into `data/` (network; needs `SYNAPSE_AUTH_TOKEN`). |
| `python scripts/01_audit_sage.py` | Read-only data audit (schema, integrity, split leakage, labels, metadata, text). Writes tables, figures and `AUDIT_REPORT.md` to `outputs/audit/`. No network. Exits non-zero if train and test share a patient or visit. |
| `python scripts/03_build_manifests.py` | Per-image manifests for SAGE and HyperKvasir in the six-class label space (`src/sage_shift/labelspace.py`). Writes to `outputs/manifests/`, or to `--out-dir` (a private Google Drive folder on Colab). No network. Exits non-zero if any count differs from `configs/experiment.json`. |
| `python scripts/04_constant_baselines.py` | Scores of constant predictors on the SAGE six-class test set, compared with the paper's Table 6. Writes to `outputs/baselines/`. |
| `python scripts/05_check_fov.py` | Field-of-view detection, field layouts, quality flags and the Reinhard colour reference (from HyperKvasir), plus contact sheets for a visual check. Writes `preprocess_reference.json` (aggregate) and per-image quality tables to `outputs/preprocess/`. CPU, no network. On Colab, run with `--no-sheets`. |
| `python scripts/06_extract_features.py` | Frozen-backbone features for the raw, fov, colour and fov_colour variants, and for the exploratory inputs (SAGE: masked, masked_grey, letterbox, letterbox_grey, fov_grey, colour_global, fov_colour_global; HyperKvasir: sageify, sageify_grey; `--colour-global` gives the SAGE colour statistics from script 16). Writes `<dataset>/<backbone>/<variant>.npz` and a run record to `outputs/features/` or `--out-dir`. Meant for a GPU (e.g. Google Colab); downloads pretrained weights on first use. `--random-weights --limit 8` is an offline smoke test. |
| `python scripts/07_rq1_probes.py` | Logistic-regression probes on the features: HyperKvasir-trained (transfer) and SAGE-trained (the SAGE-only reference, named `ceiling` in files), scored on the SAGE six-class test set, with paired patient-level bootstrap CIs for the gap. Writes aggregate tables to `outputs/rq1/` and per-image predictions to `outputs/rq1/predictions/`. CPU. |
| `python scripts/08_paper_recipe.py --recipe faithful` | Re-runs the SAGE paper's HyperKvasir training recipe (ResNet-50 fine-tuning) as released (`faithful`) or with a lower learning rate and the official folds (`adjusted`), and scores the chosen epoch on SAGE. With `--reference`, also scores fov crops of the SAGE frames. Resumes after an interruption. Needs a GPU; `--random-weights --limit 8 --device cpu` is a smoke test. |
| `python scripts/09_rq2_analysis.py` | RQ2 tables from the saved predictions: the gap per variant and with EM prior correction, per-class gaps by meaning group, polyp context, sensitivity and open-set checks, a probe trained on older-format HyperKvasir frames only, and the paper-recipe re-runs. Paired patient-level bootstrap throughout. Writes aggregate tables to `outputs/rq2/`. CPU. |
| `python scripts/10_rq3_label_efficiency.py` | RQ3 label-efficiency curves: probes trained on k SAGE patients (seeded draws), with and without HyperKvasir, reusing the RQ1 C values; reports the smallest k that reaches 90% of the SAGE-only reference. Writes aggregate tables to `outputs/rq3/`. CPU. |
| `python scripts/11_figures.py` | Figures from the tables above (confusion matrices, per-class gaps, label-efficiency curves, re-runs on raw, masked and fov-cropped frames, a conceptual diagram of the shift factors, and leave-one-patient-out influence once script 12 has run; macro-AUROC curves are added once script 13 has run) as PNG and PDF in `outputs/figures/`. Figures with dataset images are never made. CPU. |
| `python scripts/12_sensitivity.py` | Sensitivity analyses of the RQ1–RQ2 results: patient-weighted metrics, BCa and studentized intervals, patient-level swap tests, multiplicity adjustments, effective numbers of patients, power, MCC and balanced accuracy, leave-one-patient-out influence, polyp-context intervals, training variability and numerical robustness. Writes aggregate tables to `outputs/sensitivity/`. CPU. |
| `python scripts/13_exploratory_probes.py` | Exploratory probe analyses: probes trained on raw HyperKvasir scored on processed SAGE inputs, prior-correction diagnostics with calibration and an oracle prior, regrouped class groups, a source without esophagitis grade A, RQ3 AUROC curves, ratios and regularisation sensitivity, intercept-only recalibration, and SAGE field layout by location. Writes aggregate tables to `outputs/exploratory/`. CPU. |
| `python scripts/14_geometry_controls.py` | Scores kept re-run weights (script 08 with `--keep-model`) on SAGE test frames under raw, masked, masked_grey, fov, fov_grey, letterbox and letterbox_grey inputs, and on HyperKvasir fold 1 under raw, sageify and sageify_grey. Needs a GPU; has a CPU smoke mode. |
| `python scripts/15_geometry_analysis.py` | Analysis of the geometry and colour inputs: frozen-probe and fine-tuned results with paired intervals, the HyperKvasir counterfactual, and the stated reading rules. Writes aggregate tables to `outputs/exploratory_gpu/`. CPU. |
| `python scripts/16_colour_global_stats.py` | Dataset-level Lab mean and SD of SAGE train frames (from the field-quality table of script 05), used by the colour_global inputs of script 06; run it before those inputs. CPU. |

Tests (standard library only; the feature tests are skipped without PyTorch): `python -m unittest discover tests`.

## Reproducibility notes

- Feature extraction and the re-runs were done on Google Colab (Tesla T4) with Python 3.13, PyTorch 2.14.1+cu130 and torchvision 0.29.1+cu130, as recorded in each run record. The versions pinned in `requirements-torch.txt` are for CPU smoke tests.
- DINOv2 is loaded from the `facebookresearch/dinov2` torch.hub repository (main branch); the run record stores the SHA-256 of the hub code and of the weights.
- GPU runs are not bit-for-bit repeatable (cuDNN non-determinism).
- Logistic-regression results can differ in the last digits with the number of BLAS threads; the saved results used all available threads on one machine. Script 12 reports the size of this effect.

## HyperKvasir

HyperKvasir (Borgli et al., *Scientific Data* 2020; CC BY 4.0) is downloaded separately into `external/hyperkvasir/` (git-ignored):

```
external/hyperkvasir/
├── official_splits/   2_fold_split.csv, 5_fold_split.csv (github.com/simula/hyper-kvasir)
└── labeled-images/    the extracted labelled-images download (image-labels.csv + image folders)
```

## Outputs

Manifests, features and per-image predictions are per-image data. They are written to `outputs/` (git-ignored) or, on Colab, to a mounted private Google Drive folder passed as `--out-dir`, and are never committed.

## Layout

```
src/sage_shift/   Python package with the project logic
scripts/          numbered entry-point scripts
configs/          configuration files
tests/            unit tests (python -m unittest discover tests)
notebooks/        Colab notebooks (later)
data/             SAGE dataset (git-ignored)
outputs/          generated tables, figures and reports (git-ignored)
external/         HyperKvasir and other external datasets (git-ignored)
```
