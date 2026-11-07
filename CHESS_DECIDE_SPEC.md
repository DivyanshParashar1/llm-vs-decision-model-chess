# Chess Decision-Model Experiment: Project Spec & Starter Guide

> **Audience:** a coding agent (or engineer) picking this project up cold.
> **Goal of this doc:** explain the idea, the experiment design, how data is collected and labeled, what to build, and in what order, so work can start without further context.

---

## 1. The idea in one paragraph

Fine-tuning an ~8B LLM is expensive. A **small encoder "decision model"** (GLiNER2.5-Decide, 340M params) is cheap to fine-tune and fast at inference. The hypothesis:

> **An off-the-shelf LLM that proposes options, plus a small fine-tuned decision model that picks among them, can match or beat a fully fine-tuned LLM, at a fraction of the training cost.**

Chess is the testbed because answers are **objectively scorable** (Stockfish gives ground truth), data is **free and unlimited** (Lichess), and the task naturally splits into *generate candidates* (LLM) and *choose one* (decision model).

---

## 2. Background: what GLiNER2.5-Decide is

- Released by Fastino Labs on 24 Sep 2026. Apache-2.0. HF: `fastino/GLiNER2.5-Decide` (340M, DeBERTa-v3-large encoder). Siblings: `fastino/GLiNER2.5-Decide-1B`, `GLiNER2.5-multi-Decide`.
- **Input:** a text plus a **caller-supplied list of labels** (optionally with descriptions). **Output:** a probability per label (single-label, multi-label, or ordinal score). No generated tokens. Deterministic.
- Install / load:
  ```bash
  pip install gliner2
  ```
  ```python
  from gliner2 import AutoExtractor
  model = AutoExtractor.from_pretrained("fastino/GLiNER2.5-Decide")
  ```
- It does **not** reason or generate. That's why it's paired with an LLM: the LLM supplies the candidates and the context; GLiNER makes the call.

> ⚠️ **Verify before building:** (a) the exact classification call signature in `gliner2` (method name, how labels/descriptions are passed, how probabilities come back); (b) the **max input length** (DeBERTa encoders are often ~512 tokens, so the prompt must fit); (c) the **max practical number of labels** per call; (d) the supported **fine-tuning path** (local trainer in `gliner2` vs. Fastino hosted fine-tuning API). Write a tiny `scripts/smoke_gliner.py` that answers all four and record the answers in `docs/gliner_notes.md`.

---

## 3. The five experimental configurations

Notation: **LLM₀** = base instruct LLM, **LLM\*** = fine-tuned LLM, **D₀** = base GLiNER2.5-Decide, **D\*** = fine-tuned GLiNER2.5-Decide.

| ID | Config | What it answers |
|----|--------|-----------------|
| **C1** | **LLM₀ alone** | Baseline: how good is a generic LLM at chess? |
| **C2** | **LLM₀ → D₀** (base decider picks from LLM₀'s candidates) | Does an *untrained* decider add anything? (zero-shot control) |
| **C3** | **LLM\* alone** (LLM fine-tuned on chess) | The expensive approach we want to match. |
| **C4** | **LLM₀ → D\*** (fine-tuned decider over base LLM) | **The core claim.** Can cheap decider fine-tuning replace LLM fine-tuning? |
| **C5** | **LLM\* → D\*** (fine-tuned decider over fine-tuned LLM) | Are the gains additive? Upper bound of the approach. |

**Headline comparison:** **C4 vs C3**, judged on both quality *and* training cost. Supporting comparisons: C4 vs C1 (decider gain), C2 vs C1 (zero-shot decider is not the reason), C5 vs C3 and C5 vs C4 (complementarity).

### Recommended extra controls (cheap, strongly recommended)
| ID | Control | Why |
|----|---------|-----|
| **X1** | **D\* over *all legal moves*, no LLM** | Shows whether the LLM is actually contributing, or whether D\* is just a chess engine on its own. |
| **X2** | **Oracle picker over LLM₀ candidates** (always choose the Stockfish-best among the k candidates) | Upper bound for C2/C4. If the oracle is weak, the LLM's candidate list is the bottleneck, not the decider. |
| **X3** | **Random picker over LLM₀ candidates** | Lower bound for C2/C4. |
| **X4** | **C5 with D\* re-trained on LLM\*'s candidates** | C5 as specified reuses D\* trained on LLM₀'s candidates (distribution shift). X4 says whether that shift matters. |

---

## 4. How the pipeline works (per position)

```
FEN ──► [Proposer: LLM]  ──► k candidate moves (+ short rationale each)
                                   │
                                   ▼
          [Decider: GLiNER2.5-Decide]  labels = the k candidate moves
                                   │
                                   ▼
                          chosen move (UCI)
```

### 4.1 Proposer (LLM) contract
- **Input:** FEN, side to move, list of **legal moves in SAN** (giving the legal list sharply cuts illegal outputs; keep this identical for C1/C3 so the comparison is fair).
- **Output (JSON):** `{"candidates": [{"move": "Nf3", "why": "..."} , ...]}` with **k = 5** (configurable). Parse with python-chess; drop illegal/duplicate moves; if fewer than k remain, **pad with legal moves** by a fixed deterministic rule (e.g. Stockfish-free heuristic: captures/checks first, then alphabetical) and **log the pad count**.
- **Single-model configs (C1, C3):** the same LLM is asked for **one** move (`{"move": "..."}`). Alternatively take candidate #1 from the k-list. **Pick one and use it everywhere**; recommended: separate single-move prompt, since that is how a "raw LLM" would be used.
- Decoding: temperature 0 (greedy), fixed seed, fixed max tokens. Log raw output.

### 4.2 Decider (GLiNER) contract
- **Text input** (must fit the encoder context): FEN, an ASCII board (optional, if it fits), side to move, and the LLM's rationales.
- **Labels:** the k candidate moves in SAN, optionally with the LLM's rationale as the label *description*.
- **Output:** probability per candidate → argmax = chosen move.
- Keep a single `render_decider_input(position, candidates)` function, versioned, used identically in training and eval.

### 4.3 Failure policy (decide once, apply to all configs)
- **Illegal / unparseable move** from the LLM in C1/C3: counts as a **failure** in accuracy metrics. For game-play eval, substitute a deterministic fallback legal move and count it.
- Report **legal-move rate** separately for every config.

---

## 5. Data

### 5.1 Sources (all free)
| Source | What | Use |
|--------|------|-----|
| **Lichess puzzle database** (`database.lichess.org`, CSV) | ~5M puzzles: `PuzzleId, FEN, Moves, Rating, Themes, GameUrl, …` | Tactical positions with a known correct answer; stratify by rating. |
| **Lichess standard games** (monthly PGN dumps; a subset has `[%eval]` annotations) | Real game positions | Non-tactical / quiet middlegame and endgame positions. |
| **Stockfish** (local, pinned version, e.g. SF 17) | Engine evals | Ground-truth labels for *every* position and *every* candidate. |

> ⚠️ **Puzzle FEN gotcha:** in the Lichess puzzle CSV, the `FEN` is the position **before the opponent's move**. The first move in `Moves` is the opponent's; apply it first. The position to solve is *after* that move, and the solution is `Moves[1]`. Getting this wrong silently ruins the dataset.

### 5.2 Position sampling
- Target mix (configurable): **~60% puzzles** (stratified rating bands: <1200, 1200–1600, 1600–2000, 2000–2400, 2400+), **~40% game positions** (skip first 8 plies, skip positions with |eval| > 500cp, sample ≤ 3 positions per game).
- **Deduplicate** by normalized FEN (first 4 fields: board, side, castling, en passant).
- Keep `source`, `rating_band`, `phase` (opening/middlegame/endgame by material), and `themes` as metadata for sliced analysis.

### 5.3 Labeling with Stockfish
For every position:
- `best_move`, `best_eval` from Stockfish at fixed settings (**depth 18 or nodes = 1M, MultiPV = 5, 1 thread, hash fixed**). Pin these in config; record SF version.
- Store the MultiPV list (top-5 moves + evals).
- Convert cp → **win probability** with the Lichess formula `wp = 1 / (1 + exp(-0.00368208 * cp))` (mate scores → 0/1).

For every **LLM-generated candidate** (needed to train D\*):
- Evaluate the position after each candidate (or eval the candidate directly via `searchmoves`) at the same settings → `cand_eval`, `wp_loss = wp(best) - wp(cand)`.

### 5.4 Splits (leakage rules)
- Split by **source group**: puzzles by `PuzzleId`, game positions by **game** (never put two positions from one game in different splits).
- No normalized-FEN overlap between splits.
- Suggested sizes (tune to compute): **train 50k–200k**, **val 5k**, **test 5k** (fixed, frozen, hashed; never touched during tuning).
- Commit the split manifests (IDs only) so results are reproducible.

### 5.5 Training datasets
| Dataset | Built from | Example |
|---------|-----------|---------|
| **LLM SFT set** (for LLM\*) | train positions | prompt (FEN + legal moves) → target `{"move": best_move}`. Optional variant: target is a k-candidate list = Stockfish MultiPV top-k (so LLM\* can also act as a proposer in C5). **Train LLM\* on both formats** (mixed) so the same LLM\* serves C3 and C5. |
| **Decider set** (for D\*) | train positions **+ LLM₀ candidates** | text = `render_decider_input(...)`, labels = the k candidates, gold = candidate with the lowest `wp_loss`. If the true best move isn't among the candidates, **still train** on the best available candidate, but flag it (`best_in_candidates=false`). |
| **Decider set for X4** | same, but candidates from LLM\* | only if running X4 |

> Generating LLM₀ candidates for 100k+ positions is the most expensive data step. Use **vLLM** batched inference and cache outputs to disk (`data/candidates/llm0/*.parquet`), keyed by FEN hash + prompt version + model revision.

---

## 6. Fine-tuning

### 6.1 LLM\* (the "expensive" arm)
- Base: an ~8B instruct model (e.g. **Qwen3-8B** or **Llama-3.1-8B-Instruct**; pick one and pin the HF revision).
- Method: **LoRA / QLoRA SFT** (TRL `SFTTrainer` or Unsloth). Starting point: r=16, α=32, lr 1e-4, 1–2 epochs, cosine schedule, loss on assistant tokens only.
- Early-stop / checkpoint-select on **val top-1 accuracy**, not loss.

### 6.2 D\* (the "cheap" arm)
- Base: `fastino/GLiNER2.5-Decide` (340M). Full fine-tune is feasible; use the `gliner2` training path or Fastino's fine-tune API (see §2 verification).
- Same train positions as LLM\* (same position IDs) so the comparison is about *method*, not *data*.
- Checkpoint-select on val top-1 accuracy of the **full LLM₀ → D\*** pipeline.

### 6.3 Fairness & cost accounting (this is part of the result)
For each fine-tuned model, log: **trainable params, total params, GPU type, GPU-hours, peak VRAM, # training examples, wall-clock**. Also log the **one-off cost of generating LLM₀ candidates** for D\* training: it's a real cost of the C4 approach and must be counted honestly.

Also run a **data-scaling curve**: train both LLM\* and D\* on {5k, 20k, 50k, (200k)} positions. "C4 reaches C3 quality with N× less compute/data" is the strongest form of the claim.

---

## 7. Evaluation

### 7.1 Metrics (per config, on the frozen test set)
| Metric | Definition |
|--------|-----------|
| **Legal-move rate** | % outputs that parse to a legal move |
| **Top-1 accuracy** | chosen move == Stockfish best move |
| **Top-3 accuracy** | chosen move ∈ Stockfish MultiPV top-3 |
| **Puzzle solve rate** | first solution move correct (puzzles only) |
| **Mean win-prob loss** | mean of `wp(best) - wp(chosen)` (prefer over raw cp loss; cap cp at ±1000 if you also report ACPL) |
| **Blunder rate** | % moves with `wp_loss > 0.20` |
| **Candidate recall@k** | % positions where best move ∈ LLM's k candidates (C2, C4, C5, X2) |
| **Latency** | p50/p95 ms per move, end-to-end, on stated hardware |
| **Cost** | training GPU-hours (§6.3), inference cost per 1k moves |

**Slices:** by rating band, by phase, by source (puzzle vs game), by top themes (fork, pin, mateIn1/2, endgame…).

### 7.2 Statistics
- **95% bootstrap CIs** (10k resamples) for every metric.
- **Paired McNemar test** on top-1 correctness for the key pairs: C4 vs C3, C4 vs C1, C5 vs C3, C2 vs C1.
- Run every stochastic piece with **3 seeds** where it applies (training); report mean ± std.

### 7.3 Optional phase 2: actual play
- Play each config vs **Stockfish at fixed skill levels / UCI_Elo** (e.g. 1350, 1600, 1900), 100+ games per pairing, alternate colors, fixed opening book (e.g. 50 balanced 4-ply openings). Estimate Elo with BayesElo/Ordo.
- Do this only after offline metrics are solid; it's slow and noisy.

---

## 8. Repository layout (starter package)

```
chess-decide/
├── README.md                     # quickstart; links to this spec
├── SPEC.md                       # this document
├── pyproject.toml                # deps: python-chess, torch, transformers, peft, trl,
│                                 #       vllm, gliner2, pandas, pyarrow, zstandard,
│                                 #       hydra-core (or pydantic-settings), scipy, tqdm, wandb (optional)
├── configs/
│   ├── data.yaml                 # sources, mix, SF settings, split sizes, seeds
│   ├── llm.yaml                  # base model + revision, prompts version, decoding
│   ├── decider.yaml              # GLiNER checkpoint, k, input template version
│   ├── train_llm.yaml
│   ├── train_decider.yaml
│   └── eval.yaml                 # which configs (C1–C5, X1–X4), metrics, slices
├── src/chessdecide/
│   ├── data/
│   │   ├── lichess_puzzles.py    # download + parse CSV (handles the FEN/first-move gotcha)
│   │   ├── lichess_games.py      # stream .pgn.zst, sample positions
│   │   ├── stockfish_label.py    # MultiPV labeling, candidate eval, wp conversion
│   │   ├── splits.py             # grouped, dedup'd splits + manifests
│   │   └── build_datasets.py     # LLM SFT set, decider set
│   ├── proposer/
│   │   ├── prompts.py            # versioned prompt templates (single-move & k-candidate)
│   │   ├── llm_proposer.py       # vLLM / HF backends, JSON parsing, legality, padding
│   │   └── cache.py
│   ├── decider/
│   │   ├── render.py             # render_decider_input(): versioned, shared train/eval
│   │   ├── gliner_decider.py     # wraps gliner2 classify
│   │   └── baselines.py          # random picker, oracle picker, first-candidate
│   ├── pipelines.py              # build C1–C5, X1–X4 from components
│   ├── train/
│   │   ├── train_llm.py          # LoRA SFT
│   │   └── train_decider.py      # GLiNER fine-tune
│   ├── eval/
│   │   ├── run_eval.py           # runs configs on test set, writes per-position results
│   │   ├── metrics.py
│   │   ├── stats.py              # bootstrap, McNemar
│   │   └── play.py               # optional phase-2 games
│   └── utils/ (fen.py, logging.py, seeds.py, cost.py)
├── scripts/
│   ├── smoke_gliner.py           # §2 verification
│   ├── smoke_pipeline.py         # 20 positions end-to-end through C1–C5 with dummy models
│   └── make_report.py            # tables + plots → reports/
├── data/                         # gitignored; raw/, labeled/, candidates/, splits/
├── runs/                         # gitignored; checkpoints + logs
├── reports/                      # results tables, plots, final write-up
└── tests/
    ├── test_puzzle_parsing.py    # the FEN/first-move gotcha
    ├── test_legality.py
    ├── test_splits_no_leak.py
    ├── test_render_deterministic.py
    └── test_metrics.py
```

### Core interfaces (keep these stable)
```python
@dataclass
class Position:
    id: str; fen: str; source: str; meta: dict
    sf_best: str; sf_multipv: list[tuple[str, int]]   # (uci, cp)

class Proposer(Protocol):
    def propose(self, pos: Position, k: int) -> list[Candidate]: ...
    def single_move(self, pos: Position) -> str | None: ...

class Decider(Protocol):
    def choose(self, pos: Position, cands: list[Candidate]) -> tuple[str, dict[str, float]]: ...

class Pipeline(Protocol):
    name: str                      # "C1".."C5", "X1".."X4"
    def move(self, pos: Position) -> MoveResult: ...   # move, legal?, latency, debug info
```

Per-position results go to `runs/<run_id>/eval/<config>.parquet` with: position id, chosen move, legal, candidates, decider probs, wp_loss, latency, prompt/template versions, model revisions. **Every metric is computed from these files**, so nothing has to be re-run to add a metric.

---

## 9. Milestones (with acceptance criteria)

| # | Milestone | Done when |
|---|-----------|-----------|
| **M0** | Env + smoke tests | `smoke_gliner.py` answers the 4 questions in §2; Stockfish + python-chess + vLLM work; `pytest` green |
| **M1** | Data v1 | 10k labeled positions with splits + manifests; puzzle parsing test passes; no-leak test passes |
| **M2** | Baselines | C1, C2, X2, X3 run on a 1k dev slice; recall@k reported. **Checkpoint: if X2 (oracle) ≈ C1, the candidate list is the bottleneck → raise k or improve the proposer prompt before going further.** |
| **M3** | Candidate generation at scale | LLM₀ candidates cached for the full train set |
| **M4** | Fine-tune D\* | C4 beats C1 on val with CI not overlapping |
| **M5** | Fine-tune LLM\* | C3 trained with equal data; C5 runs |
| **M6** | Full eval | All configs on frozen test; stats; slices; cost table; scaling curve |
| **M7** | Report | `reports/RESULTS.md` with headline table, C4-vs-C3 verdict, cost comparison, failure analysis, limitations |

Start with **small scale** (k=5, 10k train, 1k test) end-to-end before scaling anything.

---

## 10. Risks & things to watch

- **Encoder context length:** FEN + board + 5 rationales may exceed it. Drop the ASCII board first, then truncate rationales. Measure truncation rate.
- **Candidate bottleneck:** D\* can never beat the oracle (X2). Report recall@k prominently.
- **Distribution shift in C5:** D\* trained on LLM₀ candidates; LLM\* candidates look different. X4 quantifies it.
- **Is D\* just memorizing chess?** X1 answers this. If X1 ≈ C4, the story becomes "small encoder can play chess", which is still interesting but a different claim.
- **Prompt sensitivity of C1:** a bad prompt makes the baseline look artificially weak. Do a small prompt search (≤5 prompts) for LLM₀ on val **before** comparing, and report it.
- **Stockfish settings drift:** pin version, depth/nodes, threads, hash. Labels must be reproducible.
- **Leakage via Lichess puzzle source games:** puzzles share `GameUrl` with game dumps; exclude puzzle source games from the game-position pool or keep them in the same split.

---

## 11. Open decisions for the project owner

1. Which base LLM (Qwen3-8B vs Llama-3.1-8B vs other)?
2. GLiNER2.5-Decide **340M** (default) or **1B**? Run 1B as an ablation only if 340M looks promising.
3. k (number of candidates): default 5; try 3 and 8 in ablation?
4. Compute budget (GPU type and hours) → sets train size and whether the scaling curve and phase-2 play are in scope.
5. Local GLiNER fine-tuning vs Fastino hosted fine-tuning.

---

## 12. Glossary
- **FEN:** one-line board-state encoding. **SAN/UCI:** human (`Nf3`) / engine (`g1f3`) move notation.
- **MultiPV:** Stockfish returns the top-N moves, not just the best.
- **cp:** centipawns (100 = one pawn). **wp:** win probability derived from cp.
- **Proposer / Decider:** the LLM that generates candidate moves / the GLiNER model that picks one.
