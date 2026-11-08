# LLM vs Decision Model for Chess

Can a small fine-tuned **decision model** (GLiNER2.5-Decide, advertised as 340M parameters) picking among an
off-the-shelf LLM's candidate moves match a **fine-tuned 8B LLM** (Qwen3-8B), at a fraction of the training cost?

Full design: [CHESS_DECIDE_SPEC.md](CHESS_DECIDE_SPEC.md) · Full results: [reports/RESULTS.md](reports/RESULTS.md)

## Key findings

**A small decision model, used on its own, nearly matches a fine-tuned 8B LLM.**
GLiNER2.5-Decide (advertised 340M; 486M parameters as loaded, including task heads), fine-tuned for 1.6 GPU-hours
and choosing among all legal moves with no LLM involved, **ties a QLoRA fine-tuned Qwen3-8B on move quality**:
same win-probability loss (0.261 vs 0.261) and blunder rate (44.8% vs 44.2%), and 3.5 points lower top-1 accuracy.
That is a model about 17× smaller, trained with about a quarter of the GPU-hours.

Frozen test set: 2,003 positions (60% Lichess puzzles, 40% real-game positions); ground truth from Stockfish 17.1.

| Config | Top-1 % [95% CI] | Win-prob loss ↓ | Blunder % ↓ | Training GPU-h |
|---|---|---|---|---|
| C1 Qwen3-8B (raw) | 19.5 [17.7, 21.2] | 0.393 | 66.0 | 0 |
| C4 Qwen3-8B (raw) → GLiNER fine-tuned | 26.8 [24.9, 28.8] | 0.313 | 52.6 | 1.6 (+ ~2.2 h to generate the LLM candidates) |
| **C3 Qwen3-8B fine-tuned (QLoRA)** | **36.1 [34.0, 38.2]** | **0.261** | **44.2** | **6.1** |
| **X1 GLiNER fine-tuned alone, all legal moves** | **32.6 [30.7, 34.7]** | **0.261** | **44.8** | **1.6** |
| X2 oracle: best of raw LLM's 5 candidates | 46.4 | 0.163 | 29.6 | n/a |

1. **The original hypothesis is rejected.** Raw LLM + fine-tuned decider (C4) loses clearly to the fine-tuned LLM (C3):
   26.8% vs 36.1% top-1 (McNemar p ≈ 4e-17). It still improves on the raw LLM by +7.3 pts and cuts blunders by 13 pts.
2. **The LLM's candidate list was the bottleneck, not the decider.** The best move was among the raw LLM's top 5
   only 47% of the time. The same decider does better picking from *all* legal moves (X1, 32.6%) than from the LLM's shortlist.
3. **The small model generalizes as well as the big one.** On 509 puzzles from themes never seen in training, X1 and the
   fine-tuned LLM are statistically tied (46.6% vs 48.9%, p = 0.29).
4. **It is not a notation shortcut.** Mate-in-one positions are 5% of the test set and both models solve all of them;
   X1 stays close on the remaining 95%.
5. **Stacking is not additive.** The decider on top of the fine-tuned LLM (C5, 33.1%) is worse than the fine-tuned LLM alone.
6. **More decider data did not help.** 5k, 10k and 20k training positions all reach about 25% val top-1, so the decider
   saturates early (0.4 GPU-hours at 5k).
7. **Real-game positions are much harder than puzzles** for every method (fine-tuned LLM: 20% vs 47% top-1).

Limitations: one seed per run, one LLM, chess only, no hyperparameter search. The decider was trained only on the LLM's
5 candidates, never on full legal-move lists, so X1 is likely a lower bound. Details and follow-ups in
[reports/RESULTS.md](reports/RESULTS.md).

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
