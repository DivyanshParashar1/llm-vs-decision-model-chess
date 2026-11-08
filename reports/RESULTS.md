# Results: decision model vs fine-tuned LLM for chess move selection

**Question.** Can a small, cheaply fine-tuned decision model (GLiNER2.5-Decide, 486M params) choosing among an
off-the-shelf LLM's candidate moves match a fine-tuned 8B LLM (Qwen3-8B + QLoRA), at a fraction of the training cost?

**Short answer.** Not in the proposed form. The fine-tuned LLM (C3) clearly beats the raw-LLM + fine-tuned-decider
pipeline (C4): **36.1% vs 26.8% top-1** on the frozen test set (p ≈ 4e-17). The decider does add value over the raw LLM
(+7.3 pts top-1, −13 pts blunders), but it is capped by the LLM's candidate list.

**The surprise.** The same fine-tuned decider used **without the LLM**, choosing among *all* legal moves (control X1),
reaches **32.6% top-1** and **matches the fine-tuned LLM on win-probability loss (0.261 vs 0.261) and blunder rate
(44.8% vs 44.2%)**, after 1.6 GPU-hours of training versus 6.1 for the LLM. On held-out puzzle themes it is
statistically tied with the fine-tuned LLM. The LLM's candidate list was the bottleneck, not the decider.

All numbers below come from `reports/RESULTS_TABLES.md` (generated from notebook 06's `results.json`).

## Setup

- **Data** (notebook 01): 25,424 positions from Lichess, 60% puzzles (rating-stratified; multi-move puzzles split into
  one position per solver move) and 40% quiet positions from real rated games (≤3 per game, |eval| ≤ 500 cp).
  Splits are grouped by puzzle/game with no shared positions: train 19,894 · val 2,002 · test 2,003 · test_ood 509
  (three puzzle themes never seen in training: smothered mate, interference, x-ray).
  Split hashes are frozen in `docs/data_manifest.json`.
- **Ground truth**: Stockfish 17.1, MultiPV 5, 200k nodes, 1 thread, fresh hash per position (reproducible).
  Sanity check: Lichess puzzle solution == Stockfish best move in 99.7% of puzzles.
- **LLM**: Qwen3-8B (thinking off), greedy decoding, prompt = FEN + ASCII board + legal moves (SAN).
  LLM₀ = base model, LLM\* = QLoRA r16 fine-tune on Stockfish targets (1 epoch, 19,894 positions).
- **Decider**: GLiNER2.5-Decide. Input = FEN + ASCII board; labels = candidate moves with the LLM's one-line rationale
  as description. D₀ = base, D\* = full fine-tune on the same 19,894 train positions (gold = Stockfish-best candidate).
- **Hardware**: free Kaggle Tesla T4s only.

## Main results (test, n = 2,003)

| Config | Top-1 % [95% CI] | Win-prob loss | Blunder % | Full-line puzzle solve % |
|---|---|---|---|---|
| C1 raw LLM | 19.5 [17.7, 21.2] | 0.393 | 66.0 | 9.2 |
| C2 raw LLM → base decider | 17.9 [16.2, 19.6] | 0.395 | 66.0 | 8.0 |
| **C3 fine-tuned LLM** | **36.1 [34.0, 38.2]** | **0.261** | **44.2** | **21.7** |
| C4 raw LLM → fine-tuned decider | 26.8 [24.9, 28.8] | 0.313 | 52.6 | 13.1 |
| C5 fine-tuned LLM → fine-tuned decider | 33.1 [31.1, 35.1] | 0.269 | 45.7 | 20.1 |
| **X1 fine-tuned decider, all legal moves (no LLM)** | **32.6 [30.7, 34.7]** | **0.261** | **44.8** | **21.7** |
| X2 oracle over raw-LLM candidates | 46.4 | 0.163 | 29.6 | 32.1 |
| X3 random raw-LLM candidate | 9.5 | 0.449 | 72.9 | 2.0 |

Recall@5 (Stockfish-best move among the candidates): raw LLM 46.9%, fine-tuned LLM 57.6%.

Paired McNemar tests on top-1: C4 vs C3 p = 3.6e-17 · C4 vs C1 p = 3.1e-14 · C5 vs C3 p = 3.1e-4 ·
X1 vs C3 p = 0.0013 · X1 vs C4 p = 6.5e-9 · C2 vs C1 p = 0.051.

## Findings

1. **The core claim (C4 ≈ C3) is rejected.** C3 leads by 9.3 pts top-1 and has lower win-prob loss and fewer blunders.
   C4 cannot exceed the oracle over the raw LLM's candidates (46.4%), and in practice reaches about 58% of that ceiling.
2. **Fine-tuning the decider is necessary and helps.** The base decider adds nothing (C2 ≈ C1). After fine-tuning,
   C4 beats C1 by +7.3 pts top-1 and cuts blunders by 13.4 pts. On val, D\* picks the best of the 5 candidates
   43.5% of the time (base: 29.4%, random: 20%).
3. **More decider data did not help.** D\* trained on 5k, 10k and 19.9k positions scored 25.4 / 24.4 / 25.5% val top-1.
   It saturates by 5k examples (0.4 GPU-hours), which points to an input/representation limit, not a data limit.
4. **The decider alone nearly matches the fine-tuned LLM.** X1 (D\* over all legal moves) beats C4 by 5.8 pts and ties
   C3 on win-prob loss and blunder rate; C3 keeps a 3.5-pt top-1 edge (significant on test, p = 0.0013; not significant
   on test_ood, 46.6 vs 48.9, p = 0.29). This is not a mate-notation trick: positions whose best move gives mate are
   only 5.3% of test, both models solve 100% of them, and on the rest X1 scores 28.9% vs C3's 32.5%.
   D\* was never trained on full legal-move lists (only on 5 LLM candidates), so this is a lower bound for that setup.
5. **Stacking is not additive.** C5 (decider over the fine-tuned LLM's candidates) is *worse* than C3 alone
   (33.1 vs 36.1, p = 3e-4). D\* was trained on raw-LLM candidates and overrides the stronger model's first choice.
6. **Generalization to unseen themes** (test_ood): same ranking. C3 48.9%, C5 46.9%, X1 46.6%, C4 33.2%, C1 27.9%.
7. **Real-game positions are much harder than puzzles for every method** (C3: 20.2% vs 46.7%; X1: 16.3% vs 43.5%).

## Cost

| Arm | Trainable params | GPU-hours to train | Other one-off cost |
|---|---|---|---|
| LLM\* (C3) | 43.6M (LoRA) of 8.2B | **6.11** | none |
| D\* at 19.9k (C4, X1) | 486M (full) | **1.59** | raw-LLM candidates for train positions: ≈ 2.2 h of a 2.7 h 2×T4 session |
| D\* at 5k (same accuracy) | 486M (full) | **0.40** | ≈ 0.55 h of candidate generation |

Inference: the decider takes about 100–200 ms per position on a T4. C4 still needs a full LLM generation per move;
X1 needs no LLM at all.

**Cost/quality summary.** C4 is about 1.6× cheaper to build than C3 (≈ 3.8 vs 6.1 GPU-h) but clearly worse.
X1 matches C3 on move quality (win-prob loss, blunders) at ≈ 26% of C3's training GPU-hours if the candidate-generation
step is counted, and less if D\* were trained directly on legal moves (which needs no LLM at all).

## Limitations

- One seed per training run; one base LLM (Qwen3-8B); one decider size; one domain (chess move selection).
- LLM\* trained for one epoch with default QLoRA settings and no hyperparameter search; the decider likewise.
- The prompt includes the legal-move list in SAN, whose `+`/`#` markers give both models free check/mate detection.
- "Top-1" counts only the exact Stockfish move; win-prob loss is the fairer quality measure and is reported alongside.
- Win-prob loss uses Stockfish at 200k nodes; separately searched moves can score slightly differently from MultiPV lines
  (losses are clipped at 0).
- Phase-2 play against Stockfish (Elo) was out of scope for the weekend budget.

## Suggested follow-ups (cheap)

1. Train D\* directly on **all legal moves** (no LLM candidates). This tests the X1 finding properly and removes the
   candidate-generation cost.
2. Give the decider move consequences (captures, checks, the position after the move) as label descriptions.
3. GLiNER2.5-Decide-1B as the decider; a learning-rate sweep (D\* training loss only fell from 16.9 to 15.5).
4. Retrain D\* on the fine-tuned LLM's candidates (control X4) to see whether C5 becomes additive.
