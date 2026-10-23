# Artificial Societies — learned survey personas

Build a shared response model from Twin-2K-500 and construct 100 personas from permitted human survey answers. The planned application compares immediate energy-bill relief with home-energy improvements.

## Current milestone

Step 1: project scaffolding, isolated Python environment, fixed seed and executable environment-check notebook. Data downloading, cleaning, split creation and model training are later steps. The encoder revision must be pinned before feature generation.

## Start here

From this project folder:

```sh
UV_CACHE_DIR="$PWD/.cache/uv" uv sync --locked
UV_CACHE_DIR="$PWD/.cache/uv" uv run python scripts/check_environment.py
UV_CACHE_DIR="$PWD/.cache/uv" uv run jupyter lab notebooks/00_environment_check.ipynb
```

In VS Code, choose `.venv/bin/python` as the notebook interpreter. Jupyter launched through the command above uses this environment; no system-wide kernel installation is needed.

`uv.lock` records the resolved dependency versions. Python 3.12 is specified in `.python-version`. The project config sets seed `20261023`; seeds alone cannot guarantee identical accelerator results across different hardware.

## Folder structure

```text
configs/                  Experiment and environment configuration
data/raw/                 Original downloads, kept unchanged
data/interim/             Intermediate parsing outputs
data/processed/           Clean training data and permitted persona evidence
data/heldout/             Held-out labels, kept out of training notebooks
data/splits/              Saved participant/question split manifests
notebooks/                Inspection and walkthrough notebooks
src/as_personas/          Reusable Python modules
scripts/                  Repeatable command-line entry points
artifacts/features/       Cached question/answer representations
artifacts/models/         Trained model and feature-transform checkpoints
artifacts/personas/       Final 100 persona records
artifacts/logs/           Training and validation logs
outputs/figures/           Charts for analysis and presentation
outputs/reports/           Environment checks and evaluation reports
.venv/                    Isolated Python environment
.cache/                   Local package and model caches
```

Raw/processed human data, trained artifacts, local caches and generated outputs are excluded by `.gitignore`. Configuration, code, notebooks and the dependency lockfile can be version-controlled. No remote repository has been configured.

## Next milestone

Download the official question catalogue and response tables; record their revision, then save respondent and question splits before exploring answer distributions. The existing data-split proposal is not yet applied here.
