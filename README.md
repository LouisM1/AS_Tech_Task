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
outputs/          Audit reports and question–answer embeddings
```

The notebooks are self-contained; there is no separate Python package or duplicate script entry point. Add reusable model code when training actually needs it. Generated features, models, personas and charts can go under `outputs/` when they exist. `.venv/` and `.cache/` hold local tooling and downloads; they are not project source files.

## Current state

The dataset is downloaded and the splits are saved. `01_download_and_splits.ipynb` contains the full download/split logic and seed `20261023`. Running it again verifies existing source hashes and reproduces the existing partitions without silently replacing a split.

`02_question_bank.ipynb` builds `data/processed/question_bank.json`: 441 items with original wording, carried instructions, answer mappings and source records. Use each item's `question_text` for model input; `sources` is for provenance. Checks are saved in `outputs/question_bank_audit.json`.

`03_clean_responses.ipynb` joins permitted responses to the question bank and verifies them against the original labelled source. It saves `shared_training_long.parquet`, `development_anchors_long.parquet` and `final_anchors_long.parquet` in `data/processed/`. Each row contains the person/item IDs, full question text, original response code, zero-based answer index and selected answer text. Missing rows remain null; filter `is_missing == False` before fitting. Checks are saved in `outputs/response_cleaning_audit.json`.

`04_data_quality_and_leakage.ipynb` verifies source hashes, participant/question splits, wording and option mappings, clean response tables, held-out file boundaries and potential question overlap. Results and item-level review notes are saved in `outputs/data_quality_leakage_audit.json`. It reads held-out participant IDs and file metadata only, never answer columns. Related predictors remain permitted; no equivalent held-out question was identified in the assistant text review.

`05_question_answer_embeddings.ipynb` loads frozen [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) at the revision in `configs/embedding_model.json`. It encodes the complete question context, all options and each candidate answer into 1,877 normalized 1,024-dimensional vectors. No human response tables are loaded.

- `outputs/question_answer_embeddings.npy`: float32 matrix, one row per question/option pair.
- `outputs/question_answer_index.json`: row numbers, item IDs, answer codes/indices, exact input texts and token counts.
- `outputs/question_answer_embedding_audit.json`: model provenance, checksums, execution settings and validation checks.

The vectors and readable index are included in Git. Downloaded model weights remain in the ignored `.cache/` directory and can be retrieved using the pinned config. To retrieve a question, filter the index by `item_id`, sort by `answer_index`, and select those `embedding_row` values from the matrix; the notebook includes an example.

`06_embedding_checks.ipynb` independently reloads the saved features, verifies all 1,877 option lookups, joins all 749,601 observed permitted responses, and reproduces 31 vectors using direct last-token pooling. The fixed training-only paraphrase diagnostic retrieves 9/10 intended options. It narrowly confuses returning all $12 with keeping all $12 in one trust-game example; this is a semantic limitation, not an index mismatch. Findings are in `outputs/embedding_retrieval_checks.json`. These checks do not establish survey prediction accuracy.

- **People:** 1,658 shared-training, 300 development, 100 final personas.
- **Questions:** 430 permitted training columns; 92 persona-evidence columns; eight validation questions; three final-test questions.
- **Source/version:** `configs/data_source.json` records the Hugging Face commit and file checksums.
- **Split design:** `configs/data_split_plan.json`; exact assignments are in `data/splits/`.
- **Checks:** `outputs/data_split_audit.json`. `outputs/environment_check.json` retains the initial setup check.

Missing responses remain missing and numeric answers retain their original one-based option positions. Held-out labels were partitioned programmatically, without displaying answers or inspecting response distributions. Raw files still contain them: training must read the permitted `data/processed/` files, not unfiltered raw tables.

Data and output reports are included in Git so reviewers can inspect the source data, fixed splits, prepared tables and checks directly. Notebooks, configuration and the dependency lockfile provide the reproducible recipe. Local environments, caches and credentials remain git-ignored.

## Next

Fit dimensionality reduction on training-question vectors only, then train the shared response model. The Qwen encoder is frozen; no response model or persona vectors have been trained yet. The audit caveats still apply: related policy rows share a survey instrument, and public-model pretraining contamination is unknown.

Source: [Twin-2K-500](https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500), Toubia et al. (2025), [paper](https://arxiv.org/abs/2505.17479). Licence: CC BY 4.0.
