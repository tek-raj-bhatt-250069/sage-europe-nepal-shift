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
| `python scripts/06_extract_features.py` | Frozen-backbone features for the raw, fov, colour and fov_colour variants. Writes `<dataset>/<backbone>/<variant>.npz` and a run record to `outputs/features/` or `--out-dir`. Meant for a GPU (e.g. Google Colab); downloads pretrained weights on first use. `--random-weights --limit 8` is an offline smoke test. |
| `python scripts/07_rq1_probes.py` | Logistic-regression probes on the features: HyperKvasir-trained (transfer) and SAGE-trained (ceiling), scored on the SAGE six-class test set, with paired patient-level bootstrap CIs for the gap. Writes aggregate tables to `outputs/rq1/` and per-image predictions to `outputs/rq1/predictions/`. CPU. |
| `python scripts/08_paper_recipe.py --recipe faithful` | Re-runs the SAGE paper's HyperKvasir training recipe (ResNet-50 fine-tuning) as released (`faithful`) or with a lower learning rate and the official folds (`adjusted`), and scores the chosen epoch on SAGE. With `--reference`, also scores fov crops of the SAGE frames. Resumes after an interruption. Needs a GPU; `--random-weights --limit 8 --device cpu` is a smoke test. |

Tests (standard library only; the feature tests are skipped without PyTorch): `python -m unittest discover tests`.

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
