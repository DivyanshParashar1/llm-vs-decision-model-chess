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

## Tests

```bash
CHESS_SKIP_INSTALL=1 uv run --python 3.11 --with chess,pandas,numpy,zstandard,requests,tqdm,pytest pytest -q tests/
```
