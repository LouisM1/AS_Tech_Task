# Artificial Societies — learned survey personas

Build a shared response model from Twin-2K-500 and construct 100 personas from permitted human survey answers. The planned application compares immediate energy-bill relief with home-energy improvements.

## Current milestone

Steps 1–3 are complete: the isolated environment is verified, the official structured dataset is downloaded at a pinned Hugging Face commit, and participant/question splits are saved. The executed notebook `notebooks/01_download_and_splits.ipynb` shows the download and split logic cell by cell. Full question-context reconstruction, long-format cleaning and model training remain. The encoder revision must be pinned before feature generation.

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

Build the context-complete question bank and cleaned long-format records. Preserve task instructions, map the original one-based answer codes correctly, and finish the semantic leakage audit before computing model features. Load training inputs from the explicit processed partitions, not the unfiltered raw response tables.


## Dataset and saved splits

Source: [Twin-2K-500 on Hugging Face](https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500), Toubia et al. (2025), [arXiv:2505.17479](https://arxiv.org/abs/2505.17479), CC BY 4.0. The seven source files include all numeric and labelled response tables, question catalogue and documentation. Supplied persona summaries and LLM simulation outputs are not model inputs.

- `configs/data_source.json`: immutable source commit, original file URLs, sizes and SHA-256 checksums.
- `configs/data_split_plan.json`: source revision, fixed seed, question lists and split rules.
- `data/raw/twin_2k_500/`: unmodified, read-only source copies; these contain held-out answers.
- `data/splits/respondent_split.json`: fixed IDs for 1,658 training people, 300 development people and 100 final personas.
- `data/splits/question_split.json`: 430 permitted training columns, 92 persona-evidence columns, eight validation questions and three final-test questions.
- `data/splits/respondent_assignments.csv` and `question_assignments.csv`: readable assignment tables.
- `data/processed/shared_training_wave1_3.parquet`: background training responses only.
- `data/processed/development_anchors_wave1_3.parquet` and `final_anchors_wave1_3.parquet`: permitted persona evidence only.
- `data/heldout/`: validation labels, final-test labels and a separate earlier-answer repeatability reference. Keep these out of model inputs.
- `outputs/reports/data_split_audit.json`: structural verification and missingness counts; no answer distributions or model scores.

Open the split notebook:

```sh
.venv/bin/jupyter lab notebooks/01_download_and_splits.ipynb
```

Run all cells to reproduce the partitioning. Existing sources are checked against their hashes, and existing manifests/partitions must match rather than being silently replaced. The notebook retains missing answers and one-based option codes. Raw data and generated partitions remain git-ignored; the two configuration files preserve the revision and rules for reproducibility. No individual answer values are displayed in the notebook outputs.
