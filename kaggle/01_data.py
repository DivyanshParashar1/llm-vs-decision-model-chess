# %% [markdown]
# # 01 — Build the chess dataset + GLiNER2.5-Decide smoke test
#
# **Kaggle settings (right panel):** Accelerator **None** (CPU only, saves GPU quota) · Internet **On**.
#
# 1. Run all cells with `QUICK = True` first (~10 min) and check the printed sanity stats.
# 2. Set `QUICK = False`, then **Save Version → "Save & Run All (Commit)"**. It runs in the
#    background (~1–1.5 h) and you can close the tab.
# 3. The output folder `/kaggle/working/chess_data/` becomes the input of notebook 02
#    (in notebook 02: *Add Input → Your Work → this notebook*).
#
# What it builds:
# * puzzle positions (Lichess puzzle DB, rating-stratified, multi-move puzzles split per move)
# * quiet game positions (streamed from a Lichess monthly dump, nothing big is stored)
# * Stockfish 17.1 labels (MultiPV 5, fixed nodes, 1 thread, fresh hash → reproducible)
# * leakage-safe splits: train / val / test (+ pilot subset of val) + test_ood (held-out themes)

# %%
import os, subprocess, sys

def sh(cmd):
    print("$", cmd)
    subprocess.run(cmd, shell=True, check=True)

if os.environ.get("CHESS_SKIP_INSTALL") is None:
    sh(f"{sys.executable} -m pip install -q chess zstandard")

# %%
QUICK = True  # True = tiny end-to-end check (~10 min). False = full build.

CFG = dict(
    seed=1234,
    # split sizes (positions)
    n_train=20_000, n_val=2_000, n_test=2_000, n_ood=500, n_pilot=500,
    puzzle_frac=0.6,                      # rest are game positions
    max_solver_moves_per_puzzle=3,
    puzzle_rating_bands=[0, 1200, 1600, 2000, 2400, 99_999],
    holdout_themes=["smotheredMate", "interference", "xRayAttack"],  # never in train/val/test
    puzzles_url="https://database.lichess.org/lichess_db_puzzle.csv.zst",
    games_url="https://database.lichess.org/standard/lichess_db_standard_rated_2025-01.pgn.zst",
    game_skip_plies=8, max_pos_per_game=3, game_min_base_seconds=180,
    game_elo_bands=[0, 1400, 1800, 2200, 99_999],
    game_max_parsed=3_000_000,            # safety stop while streaming
    max_abs_eval_cp=500,                  # drop lopsided game positions
    oversample=1.25,                      # extra game positions to survive the eval filter
    sf_url="https://github.com/official-stockfish/Stockfish/releases/download/sf_17.1/stockfish-ubuntu-x86-64-avx2.tar",
    sf_nodes=300_000, sf_multipv=5, sf_hash_mb=64,
    gliner_model="fastino/GLiNER2.5-Decide",
    run_gliner_smoke=True,
)
if QUICK:
    for k in ["n_train", "n_val", "n_test", "n_ood", "n_pilot"]:
        CFG[k] = max(20, CFG[k] // 40)
    CFG["sf_nodes"] = 50_000

OUT = "/kaggle/working/chess_data" if os.path.exists("/kaggle") else os.path.abspath("chess_data")
RAW = "/kaggle/tmp" if os.path.exists("/kaggle") else os.path.abspath("raw_tmp")  # not saved to output
os.makedirs(OUT, exist_ok=True); os.makedirs(RAW, exist_ok=True)
print(CFG, OUT, sep="\n")

# %%
import io, re, json, math, time, glob, hashlib, inspect, pkgutil, random, tarfile
from collections import Counter
from multiprocessing import Pool

import numpy as np
import pandas as pd
import requests
import zstandard
import chess, chess.pgn, chess.engine
from tqdm.auto import tqdm

# ---------------------------------------------------------------- helpers
def norm_fen(fen: str) -> str:
    """board, side, castling, en-passant (drop move counters) — used for dedup."""
    return " ".join(fen.split()[:4])

PIECE_VAL = {chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}

def nonpawn_material(board: chess.Board) -> int:
    return sum(len(board.pieces(pt, c)) * v for pt, v in PIECE_VAL.items() for c in (chess.WHITE, chess.BLACK))

def phase_of(board: chess.Board) -> str:
    m = nonpawn_material(board)          # 62 at the start
    if m <= 26:
        return "endgame"
    if board.fullmove_number <= 12 and m >= 50:
        return "opening"
    return "middlegame"

def win_prob(cp, mate=None) -> float:
    """Lichess cp -> win probability (side to move)."""
    if mate is not None:
        return 1.0 if mate > 0 else 0.0
    return 1.0 / (1.0 + math.exp(-0.00368208 * cp))

def band_label(x, edges):
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo <= x < hi:
            return f"{lo}-{hi}" if hi < 99_999 else f"{lo}+"
    return "unknown"

def download(url, dest):
    if os.path.exists(dest):
        return dest
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest + ".part", "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=os.path.basename(dest)) as bar:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk); bar.update(len(chunk))
    os.rename(dest + ".part", dest)
    return dest

# ---------------------------------------------------------------- puzzles
def explode_puzzle(pid, fen, moves_str, cap):
    """Lichess gotcha: FEN is BEFORE the opponent's move. moves[0] = opponent,
    moves[1] = first solver move, moves[2] = opponent reply, ...
    Returns one record per solver move (up to `cap`)."""
    board = chess.Board(fen)
    moves = moves_str.split()
    n_solver = len(moves) // 2
    out = []
    for i, uci in enumerate(moves):
        mv = chess.Move.from_uci(uci)
        if mv not in board.legal_moves:
            raise ValueError(f"puzzle {pid}: illegal move {uci} at index {i}")
        if i % 2 == 1 and len(out) < cap:
            out.append(dict(fen=board.fen(), ref_uci=uci, ref_san=board.san(mv),
                            step=(i - 1) // 2, n_steps=n_solver))
        board.push(mv)
    return out

def sample_puzzles(df, target_positions, edges, cap, seed):
    """Equal positions per rating band; returns the chosen puzzle rows."""
    df = df.copy()
    df["n_pos"] = (df["Moves"].str.count(" ") + 1).floordiv(2).clip(upper=cap)
    df["band"] = [band_label(r, edges) for r in df["Rating"]]
    bands = sorted(df["band"].unique())
    per_band = math.ceil(target_positions / len(bands))
    picked = []
    for b in bands:
        sub = df[df["band"] == b].sample(frac=1.0, random_state=seed)
        csum = sub["n_pos"].cumsum()
        picked.append(sub[csum <= per_band * 1.02])
    return pd.concat(picked)

def puzzle_records(rows, cap, split_hint=None):
    recs = []
    for r in rows.itertuples(index=False):
        for p in explode_puzzle(r.PuzzleId, r.FEN, r.Moves, cap):
            b = chess.Board(p["fen"])
            recs.append(dict(
                id=f"pz_{r.PuzzleId}_{p['step']}", group=f"pz_{r.PuzzleId}", source="puzzle",
                fen=p["fen"], ref_uci=p["ref_uci"], ref_san=p["ref_san"],
                step=p["step"], n_steps=p["n_steps"], rating=int(r.Rating), band=r.band,
                themes=r.Themes, phase=phase_of(b), n_legal=b.legal_moves.count(),
                split_hint=split_hint))
    return recs

# ---------------------------------------------------------------- games (streamed)
HDR_RE = re.compile(r'^\[(\w+) "(.*)"\]\s*$')

def iter_game_texts(textio):
    """Split a PGN text stream into per-game strings without parsing moves."""
    buf = []
    for line in textio:
        if line.startswith("[Event ") and buf:
            yield "".join(buf); buf = []
        buf.append(line)
    if buf:
        yield "".join(buf)

def parse_headers(game_text):
    h = {}
    for line in game_text.splitlines():
        if not line.startswith("["):
            break
        m = HDR_RE.match(line)
        if m:
            h[m.group(1)] = m.group(2)
    return h

def game_ok(h, min_base):
    tc = h.get("TimeControl", "-")
    if tc == "-" or "+" not in tc:
        return False
    if int(tc.split("+")[0]) < min_base:
        return False
    if h.get("Termination") != "Normal" or h.get("Variant", "Standard") != "Standard":
        return False
    return h.get("WhiteElo", "?").isdigit() and h.get("BlackElo", "?").isdigit()

def sample_game_positions(game_text, h, gid, skip, k, rng):
    game = chess.pgn.read_game(io.StringIO(game_text))
    moves = list(game.mainline_moves())
    eligible = list(range(skip, len(moves)))       # position after p plies; moves[p] = human reply
    if len(eligible) < k * 3:
        return []
    plies = sorted(int(rng.choice(seg)) for seg in np.array_split(eligible, k) if len(seg))
    elo = (int(h["WhiteElo"]) + int(h["BlackElo"])) // 2
    board, out, want = game.board(), [], set(plies)
    for p, mv in enumerate(moves):
        if p in want and not board.is_game_over() and board.legal_moves.count() >= 2:
            out.append(dict(
                id=f"gm_{gid}_{p}", group=f"gm_{gid}", source="game", fen=board.fen(),
                ref_uci=mv.uci(), ref_san=board.san(mv), step=0, n_steps=1, rating=elo,
                band=band_label(elo, CFG["game_elo_bands"]), themes="", phase=phase_of(board),
                n_legal=board.legal_moves.count(), split_hint=None))
        board.push(mv)
    return out

def stream_games(url, target_positions, exclude_ids, seed):
    rng = np.random.default_rng(seed)
    edges = CFG["game_elo_bands"]
    n_bands = len(edges) - 1
    cap_per_band = math.ceil(target_positions / n_bands)
    got = Counter(); recs = []; parsed = 0
    r = requests.get(url, stream=True, timeout=60); r.raise_for_status()
    text = io.TextIOWrapper(zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(r.raw),
                            encoding="utf-8", errors="replace")
    bar = tqdm(total=target_positions, desc="game positions")
    for gt in iter_game_texts(text):
        parsed += 1
        if parsed > CFG["game_max_parsed"] or sum(got.values()) >= target_positions:
            break
        h = parse_headers(gt)
        if not game_ok(h, CFG["game_min_base_seconds"]):
            continue
        gid = h.get("Site", "").rsplit("/", 1)[-1]
        if not gid or gid in exclude_ids:
            continue
        band = band_label((int(h["WhiteElo"]) + int(h["BlackElo"])) // 2, edges)
        if got[band] >= cap_per_band:
            continue
        new = sample_game_positions(gt, h, gid, CFG["game_skip_plies"], CFG["max_pos_per_game"], rng)
        new = new[: cap_per_band - got[band]]
        got[band] += len(new); recs.extend(new); bar.update(len(new))
    bar.close(); r.close()
    print(f"parsed {parsed:,} games; positions per Elo band: {dict(got)}")
    return recs

# ---------------------------------------------------------------- stockfish
_ENG = None

def _sf_init(path, hash_mb):
    global _ENG
    _ENG = chess.engine.SimpleEngine.popen_uci(path)
    _ENG.configure({"Threads": 1, "Hash": hash_mb})

def _sf_label(args):
    fen, nodes, multipv = args
    board = chess.Board(fen)
    # game=fen -> python-chess sends `ucinewgame` (clears hash) -> deterministic with 1 thread + node limit
    infos = _ENG.analyse(board, chess.engine.Limit(nodes=nodes),
                         multipv=min(multipv, board.legal_moves.count()), game=fen)
    if isinstance(infos, dict):
        infos = [infos]
    lines = []
    for inf in infos:
        if not inf.get("pv"):
            continue
        mv = inf["pv"][0]; s = inf["score"].pov(board.turn)
        lines.append(dict(uci=mv.uci(), san=board.san(mv), cp=s.score(mate_score=100_000),
                          mate=s.mate(), depth=inf.get("depth")))
    return fen, lines

def get_stockfish():
    tar = download(CFG["sf_url"], os.path.join(RAW, "stockfish.tar"))
    with tarfile.open(tar) as t:
        t.extractall(RAW)
    path = [p for p in glob.glob(os.path.join(RAW, "stockfish", "stockfish*")) if os.path.isfile(p)][0]
    os.chmod(path, 0o755)
    return path

def label_all(fens, sf_path, workers):
    jobs = [(f, CFG["sf_nodes"], CFG["sf_multipv"]) for f in fens]
    res = {}
    with Pool(workers, initializer=_sf_init, initargs=(sf_path, CFG["sf_hash_mb"])) as pool:
        for fen, lines in tqdm(pool.imap_unordered(_sf_label, jobs, chunksize=8), total=len(jobs), desc="stockfish"):
            res[fen] = lines
    return res

# ---------------------------------------------------------------- splits
def assign_splits(df, targets, seed):
    """Whole groups (puzzle / game) go to one split. targets: ordered {split: n_positions}."""
    groups = df.groupby("group").size().sample(frac=1.0, random_state=seed)
    split_of, filled = {}, {s: 0 for s in targets}
    order = list(targets)
    i = 0
    for g, n in groups.items():
        while i < len(order) and filled[order[i]] >= targets[order[i]]:
            i += 1
        if i == len(order):
            split_of[g] = "unused"; continue
        split_of[g] = order[i]; filled[order[i]] += n
    return df["group"].map(split_of)

# %% [markdown]
# ## GLiNER2.5-Decide smoke test
# Answers the spec's §2 questions: API/call signature, max input length, how many labels per call,
# and whether a local fine-tuning path exists. Results → `chess_data/gliner_notes.json`.

# %%
def gliner_smoke():
    notes = {}
    try:
        sh(f'{sys.executable} -m pip install -q "gliner2[local]" || {sys.executable} -m pip install -q gliner2')
        import gliner2
        notes["version"] = getattr(gliner2, "__version__", None)
        notes["top_level"] = [n for n in dir(gliner2) if not n.startswith("_")]
        notes["submodules"] = [m.name for m in pkgutil.walk_packages(gliner2.__path__, "gliner2.")]
        model = None
        for loader in ["AutoExtractor", "GLiNER2"]:
            cls = getattr(gliner2, loader, None)
            if cls is None:
                continue
            try:
                t0 = time.time()
                model = cls.from_pretrained(CFG["gliner_model"])
                notes["loader"] = loader; notes["load_s"] = round(time.time() - t0, 1)
                break
            except Exception as e:
                notes[f"load_error_{loader}"] = repr(e)[:500]
        if model is None:
            return notes
        notes["model_class"] = type(model).__name__
        notes["model_methods"] = [m for m in dir(model) if not m.startswith("_")]
        notes["signatures"] = {m: str(inspect.signature(getattr(model, m)))
                               for m in notes["model_methods"]
                               if callable(getattr(model, m, None)) and re.search(r"classif|decid|predict|extract|train|fit", m)}
        notes["n_params_M"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 1) if hasattr(model, "parameters") else None
        # max input length
        for path in ["tokenizer", "processor.tokenizer", "encoder.tokenizer", "model.tokenizer"]:
            obj = model
            try:
                for a in path.split("."):
                    obj = getattr(obj, a)
                notes["tokenizer_path"] = path
                notes["tokenizer_model_max_length"] = getattr(obj, "model_max_length", None)
                tok = obj
                break
            except AttributeError:
                tok = None
        for name, mod in (model.named_modules() if hasattr(model, "named_modules") else []):
            cfg = getattr(mod, "config", None)
            if cfg is not None and hasattr(cfg, "max_position_embeddings"):
                notes["encoder_config"] = dict(module=name, model_type=getattr(cfg, "model_type", None),
                                               max_position_embeddings=cfg.max_position_embeddings,
                                               position_buckets=getattr(cfg, "position_buckets", None),
                                               relative_attention=getattr(cfg, "relative_attention", None))
                break

        # classification call: try the plausible signatures, record the first that works
        board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4")
        cands = ["Qxf7#", "Nf3", "d3", "Qe2", "Bb3"]
        text = (f"Chess position, white to move. FEN: {board.fen()}. "
                f"Which candidate move is best?")
        desc = {c: f"the move {c}" for c in cands}
        attempts = {
            "classify_text(dict labels, include_confidence)": lambda labels: model.classify_text(text, {"best_move": labels}, include_confidence=True),
            "classify_text(dict labels)": lambda labels: model.classify_text(text, {"best_move": labels}),
            "classify_text(dict w/ descriptions)": lambda labels: model.classify_text(text, {"best_move": {l: f"the move {l}" for l in labels}}),
            "classify_text(dict labels+multi_label)": lambda labels: model.classify_text(text, {"best_move": {"labels": labels, "multi_label": False}}),
            "classify(text, labels)": lambda labels: model.classify(text, labels),
            "predict(text, labels)": lambda labels: model.predict(text, labels),
        }
        notes["call_results"] = {}
        working = None
        for name, fn in attempts.items():
            try:
                t0 = time.time(); out = fn(cands)
                notes["call_results"][name] = dict(ok=True, ms=round(1000 * (time.time() - t0)), output=repr(out)[:800])
                working = working or fn
            except Exception as e:
                notes["call_results"][name] = dict(ok=False, error=repr(e)[:300])
        notes["descriptions_example"] = desc
        # how many labels per call (X1 needs all legal moves, ~30–50)
        if working:
            busy = chess.Board("r2q1rk1/pp2bppp/2n1bn2/2pp4/3P4/2PBPN2/PP1N1PPP/R2QK2R w KQ - 0 10")
            legal = [busy.san(m) for m in busy.legal_moves]
            notes["label_scaling"] = {}
            for n in [5, 10, 20, len(legal)]:
                try:
                    t0 = time.time(); out = working(legal[:n])
                    notes["label_scaling"][n] = dict(ok=True, ms=round(1000 * (time.time() - t0)), output=repr(out)[:300])
                except Exception as e:
                    notes["label_scaling"][n] = dict(ok=False, error=repr(e)[:300])
            # long input
            if tok is not None:
                for mult in [1, 4, 12, 30]:
                    long_text = text + " " + " ".join(["The knight on c6 defends e5."] * mult * 5)
                    n_tok = len(tok(long_text)["input_ids"])
                    try:
                        working(cands); notes.setdefault("long_input", {})[n_tok] = "ok"
                    except Exception as e:
                        notes.setdefault("long_input", {})[n_tok] = repr(e)[:200]
        # local fine-tuning path?
        notes["training_symbols"] = {}
        for m in notes["submodules"]:
            if "train" in m:
                try:
                    mod = __import__(m, fromlist=["*"])
                    for n, obj in vars(mod).items():
                        if inspect.isclass(obj) and obj.__module__ == m:
                            try:
                                notes["training_symbols"][f"{m}.{n}"] = str(inspect.signature(obj))[:400]
                            except (TypeError, ValueError):
                                notes["training_symbols"][f"{m}.{n}"] = "?"
                except Exception as e:
                    notes["training_symbols"][m] = f"import error: {e!r}"[:200]
    except Exception as e:
        notes["fatal"] = repr(e)[:800]
    return notes

if CFG["run_gliner_smoke"]:
    gl_notes = gliner_smoke()
    with open(os.path.join(OUT, "gliner_notes.json"), "w") as f:
        json.dump(gl_notes, f, indent=2, default=str)
    print(json.dumps(gl_notes, indent=2, default=str)[:6000])

# %% [markdown]
# ## Puzzles

# %%
t_start = time.time()
total = CFG["n_train"] + CFG["n_val"] + CFG["n_test"]
n_puz_target = int(total * CFG["puzzle_frac"])
n_game_target = total - n_puz_target

pz_path = download(CFG["puzzles_url"], os.path.join(RAW, "lichess_db_puzzle.csv.zst"))
pz = pd.read_csv(pz_path, usecols=["PuzzleId", "FEN", "Moves", "Rating", "Themes", "GameUrl"])
print(f"{len(pz):,} puzzles in DB")
pz["game_id"] = pz["GameUrl"].str.extract(r"lichess\.org/([A-Za-z0-9]{8})", expand=False)
puzzle_game_ids = set(pz["game_id"].dropna())        # games that spawned puzzles -> excluded from game pool
pz["Themes"] = pz["Themes"].fillna("")
ood_mask = pz["Themes"].str.contains(r"\b(?:" + "|".join(CFG["holdout_themes"]) + r")\b", regex=True)
print(f"held-out-theme puzzles: {ood_mask.sum():,}")

cap = CFG["max_solver_moves_per_puzzle"]
edges = CFG["puzzle_rating_bands"]
pz_in = sample_puzzles(pz[~ood_mask], n_puz_target * 1.05, edges, cap, CFG["seed"])
pz_ood = sample_puzzles(pz[ood_mask], CFG["n_ood"], edges, cap, CFG["seed"] + 1)
recs = puzzle_records(pz_in, cap) + puzzle_records(pz_ood, cap, split_hint="test_ood")
puzzle_lines = pd.concat([pz_in.assign(ood=False), pz_ood.assign(ood=True)])[
    ["PuzzleId", "FEN", "Moves", "Rating", "Themes", "band", "n_pos", "ood"]]
del pz
print(f"puzzle positions: {len(recs):,}")

# %% [markdown]
# ## Game positions (streamed, nothing large stored)

# %%
recs += stream_games(CFG["games_url"], int(n_game_target * CFG["oversample"]), puzzle_game_ids, CFG["seed"])
pos = pd.DataFrame(recs)
pos["nfen"] = pos["fen"].map(norm_fen)
before = len(pos)
pos = pos.drop_duplicates("nfen").reset_index(drop=True)
print(f"positions: {before:,} -> {len(pos):,} after FEN dedup")

# %% [markdown]
# ## Stockfish labels

# %%
sf_path = get_stockfish()
print(subprocess.run([sf_path, "--help"], capture_output=True, text=True).stdout[:200] or sf_path)
workers = os.cpu_count() or 4
t0 = time.time()
labels = label_all(pos["fen"].tolist(), sf_path, workers)
sf_secs = time.time() - t0
print(f"stockfish: {len(labels):,} positions in {sf_secs/60:.1f} min on {workers} workers")

pos["sf_multipv"] = pos["fen"].map(lambda f: json.dumps(labels.get(f, [])))
pos = pos[pos["sf_multipv"] != "[]"].copy()
first = pos["sf_multipv"].map(lambda s: json.loads(s)[0])
pos["sf_best_uci"] = first.map(lambda d: d["uci"])
pos["sf_best_san"] = first.map(lambda d: d["san"])
pos["sf_best_cp"] = first.map(lambda d: d["cp"])
pos["sf_best_mate"] = first.map(lambda d: d["mate"])
pos["sf_best_wp"] = first.map(lambda d: win_prob(d["cp"], d["mate"]))

# quiet game positions only
gm = pos["source"] == "game"
lopsided = gm & ((pos["sf_best_cp"].abs() > CFG["max_abs_eval_cp"]) | pos["sf_best_mate"].notna())
print(f"dropping {lopsided.sum():,} lopsided game positions")
pos = pos[~lopsided].reset_index(drop=True)

# %% [markdown]
# ## Splits (grouped by puzzle / game, separately per source so the mix is the same in every split)

# %%
pos["split"] = "unused"
ood = pos["split_hint"] == "test_ood"
pos.loc[ood, "split"] = "test_ood"
for src, frac in [("puzzle", CFG["puzzle_frac"]), ("game", 1 - CFG["puzzle_frac"])]:
    m = (pos["source"] == src) & ~ood
    targets = {"test": round(CFG["n_test"] * frac), "val": round(CFG["n_val"] * frac),
               "train": round(CFG["n_train"] * frac)}
    pos.loc[m, "split"] = assign_splits(pos[m], targets, CFG["seed"]).values

# pilot = random subset of val (go/no-go run in notebook 02)
val_idx = pos.index[pos["split"] == "val"]
rng = np.random.default_rng(CFG["seed"])
pos["pilot"] = False
pos.loc[rng.choice(val_idx, size=min(CFG["n_pilot"], len(val_idx)), replace=False), "pilot"] = True

# leakage checks
for a in ["train", "val", "test", "test_ood"]:
    for b in ["train", "val", "test", "test_ood"]:
        if a < b:
            ga = set(pos.loc[pos.split == a, "group"]); gb = set(pos.loc[pos.split == b, "group"])
            fa = set(pos.loc[pos.split == a, "nfen"]); fb = set(pos.loc[pos.split == b, "nfen"])
            assert not (ga & gb), f"group leak {a}/{b}"
            assert not (fa & fb), f"FEN leak {a}/{b}"
print("no group / FEN leakage between splits ✔")

# %% [markdown]
# ## Sanity stats + save

# %%
pzm = pos["source"] == "puzzle"
agree = (pos.loc[pzm, "ref_uci"] == pos.loc[pzm, "sf_best_uci"]).mean()
agree_in5 = pos[pzm].apply(lambda r: r.ref_uci in [d["uci"] for d in json.loads(r.sf_multipv)], axis=1).mean()
print(f"puzzle sanity — Lichess solution == SF best: {agree:.1%}, in SF top-5: {agree_in5:.1%}")
if agree < 0.7:
    print("⚠️  LOW agreement: puzzle FEN handling or SF settings are probably wrong — do not proceed.")

print(pd.crosstab(pos["split"], pos["source"], margins=True))
print(pd.crosstab(pos["split"], pos["phase"]))
print(pd.crosstab(pos.loc[pzm, "split"], pos.loc[pzm, "band"]))
print("mean legal moves:", round(pos["n_legal"].mean(), 1))

keep = pos[pos["split"] != "unused"].drop(columns=["split_hint"]).reset_index(drop=True)
keep.to_parquet(os.path.join(OUT, "positions.parquet"), index=False)
gsplit = keep.groupby("group")["split"].first()
puzzle_lines["split"] = puzzle_lines["PuzzleId"].map(lambda p: gsplit.get(f"pz_{p}", "unused"))
puzzle_lines[puzzle_lines["split"] != "unused"].to_parquet(os.path.join(OUT, "puzzle_lines.parquet"), index=False)

manifest = {}
for s, ids in keep.groupby("split")["id"]:
    ids = sorted(ids)
    manifest[s] = dict(n=len(ids), sha256=hashlib.sha256("\n".join(ids).encode()).hexdigest())
    with open(os.path.join(OUT, f"split_{s}.txt"), "w") as f:
        f.write("\n".join(ids))
meta = dict(cfg=CFG, quick=QUICK, manifest=manifest, stockfish=os.path.basename(CFG["sf_url"]),
            sf_minutes=round(sf_secs / 60, 1), sf_workers=workers,
            puzzle_sf_agreement=round(float(agree), 4), puzzle_sf_top5=round(float(agree_in5), 4),
            total_minutes=round((time.time() - t_start) / 60, 1))
with open(os.path.join(OUT, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)
print(json.dumps(meta["manifest"], indent=2))
print(f"done in {meta['total_minutes']} min → {OUT}")
print(sorted(os.listdir(OUT)))
