# LLM vs Decision Model for Chess

Can a small fine-tuned **decision model** (GLiNER2.5-Decide, 340M) picking among an off-the-shelf
LLM's candidate moves match a **fine-tuned 8B LLM** (Qwen3-8B), at a fraction of the training cost?

Full design: [CHESS_DECIDE_SPEC.md](CHESS_DECIDE_SPEC.md).

| ID | Config |
|----|--------|
| C1 | Qwen3-8B alone |
| C2 | Qwen3-8B → GLiNER (base) |
| C3 | Qwen3-8B fine-tuned alone |
| C4 | Qwen3-8B → GLiNER fine-tuned (**core claim**) |
| C5 | Qwen3-8B fine-tuned → GLiNER fine-tuned (optional ceiling) |

## Compute

Free tiers only. Priority: **Kaggle > Colab > Mac**. Every notebook is resumable and fits one Kaggle session.

## Notebooks (`kaggle/`)

Source lives in `kaggle/NN_*.py` (`# %%` cells). Build the uploadable notebook with
`python tools/py2nb.py kaggle/NN_*.py`, then on Kaggle: *File → Import Notebook*.

| # | Notebook | Kaggle accelerator | Output |
|---|----------|--------------------|--------|
| 01 | `01_data` — puzzles + streamed game positions, Stockfish labels, splits, GLiNER smoke test | None (CPU) | `chess_data/` |
| 02 | `02_llm0` — raw Qwen3-8B (vLLM): pilot metrics (C1, C2, X1–X3) + candidates for all splits | T4 ×2 | `llm0_cands.parquet`, `pilot_report.json` |
| 03 | `03_llm_star` — QLoRA fine-tune of Qwen3-8B (LLM\*) with Unsloth | T4 | `llm_star_adapter/`, `cost_llm_star.json` |
| 04 | `04_label_cands` — Stockfish-score every LLM₀ candidate, pick gold | None (CPU) | `llm0_cands_labeled.parquet`, `sf_evals.parquet` |
| 05 | `05_dstar` — fine-tune GLiNER2.5-Decide (D\*), scaling curve 5k/10k/20k | T4 | `dstar_<n>/`, `dstar_val.json`, `cost_dstar.json` |
| 06 | `06_eval` — C1–C5 + X1–X3 on test / test_ood, CIs, McNemar, slices, cost | T4 ×2 | `results.json`, `table_*.csv` |

Dependencies: 01 → (02, 03 in parallel) → 04 → 05 → 06.

Run on Kaggle from this machine (after `kaggle auth login`):

```bash
python tools/push_lib.py "msg"          # publish src/chessdecide as dataset chessdecide-lib
python tools/kpush.py kaggle/02_llm0.py --accel t4x2 --set 'MODE="gen"' \
    --input <user>/chess-01-data --dataset <user>/chessdecide-lib
```

Dataset v1 split hashes are frozen in [docs/data_manifest.json](docs/data_manifest.json); GLiNER API facts in [docs/gliner_notes.md](docs/gliner_notes.md).

## Tests

```bash
CHESS_SKIP_INSTALL=1 uv run --python 3.11 --with chess,pandas,numpy,zstandard,requests,tqdm,pytest pytest -q tests/
```
