# Artificial Societies — learned survey personas

Build 100 personas from Twin-2K-500 to explore energy-bill relief versus home-energy improvements.

## Run the project

From this folder:

```sh
UV_CACHE_DIR="$PWD/.cache/uv" uv sync --locked
.venv/bin/jupyter lab notebooks/01_download_and_splits.ipynb
```

In VS Code, select `.venv/bin/python` as the notebook interpreter. Python 3.12 and the dependency versions are recorded in `.python-version`, `pyproject.toml` and `uv.lock`.

## What is here

```text
notebooks/        The experiment, explained and executed cell by cell
configs/          Pinned data source and reproducible split rules
data/
  raw/            Unmodified source files and checksums
  processed/      Permitted training responses and persona evidence
  heldout/        Validation/test answers and the repeatability reference
  splits/         Fixed participant IDs, question roles and partition checks
outputs/          Setup and data-split reports
```

The notebooks are self-contained; there is no separate Python package or duplicate script entry point. Add reusable model code when training actually needs it. Generated features, models, personas and charts can go under `outputs/` when they exist. `.venv/` and `.cache/` hold local tooling and downloads; they are not project source files.

## Current state

The dataset is downloaded and the splits are saved. `01_download_and_splits.ipynb` contains the full download/split logic and seed `20261023`. Running it again verifies existing source hashes and reproduces the existing partitions without silently replacing a split.

- **People:** 1,658 shared-training, 300 development, 100 final personas.
- **Questions:** 430 permitted training columns; 92 persona-evidence columns; eight validation questions; three final-test questions.
- **Source/version:** `configs/data_source.json` records the Hugging Face commit and file checksums.
- **Split design:** `configs/data_split_plan.json`; exact assignments are in `data/splits/`.
- **Checks:** `outputs/data_split_audit.json`. `outputs/environment_check.json` retains the initial setup check.

Missing responses remain missing and numeric answers retain their original one-based option positions. Held-out labels were partitioned programmatically, without displaying answers or inspecting response distributions. Raw files still contain them: training must read the permitted `data/processed/` files, not unfiltered raw tables.

Data, caches and generated outputs are git-ignored. Notebooks, configuration and the dependency lockfile provide the reproducible recipe.

## Next

Build the context-complete question bank and cleaned training records, preserve task instructions and finish the semantic leakage audit. No model has been trained yet.

Source: [Twin-2K-500](https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500), Toubia et al. (2025), [paper](https://arxiv.org/abs/2505.17479). Licence: CC BY 4.0.
