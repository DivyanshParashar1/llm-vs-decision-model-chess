"""Stockfish helpers: download, and evaluate specific moves (searchmoves) in parallel.

Settings match notebook 01 (SF 17.1, 1 thread, fresh hash per search, fixed nodes) so evals are comparable.
"""
from __future__ import annotations

import glob
import math
import os
import tarfile
from multiprocessing import Pool

import chess
import chess.engine
import requests

SF_URL = "https://github.com/official-stockfish/Stockfish/releases/download/sf_17.1/stockfish-ubuntu-x86-64-avx2.tar"
NODES = 200_000


def win_prob(cp, mate=None) -> float:
    if mate is not None:
        return 1.0 if mate > 0 else 0.0
    return 1.0 / (1.0 + math.exp(-0.00368208 * cp))


def get_stockfish(workdir: str = "/kaggle/tmp") -> str:
    os.makedirs(workdir, exist_ok=True)
    tar = os.path.join(workdir, "stockfish.tar")
    if not os.path.exists(tar):
        with requests.get(SF_URL, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(tar, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
    with tarfile.open(tar) as t:
        t.extractall(workdir, filter="data")
    path = [p for p in glob.glob(os.path.join(workdir, "stockfish", "stockfish*")) if os.path.isfile(p)][0]
    os.chmod(path, 0o755)
    return path


_ENG = None


def _init(path, hash_mb=64):
    global _ENG
    _ENG = chess.engine.SimpleEngine.popen_uci(path)
    _ENG.configure({"Threads": 1, "Hash": hash_mb})


def _eval_move(job):
    """job = (fen, uci, nodes) -> (fen, uci, cp, mate) from the mover's point of view."""
    fen, uci, nodes = job
    board = chess.Board(fen)
    mv = chess.Move.from_uci(uci)
    info = _ENG.analyse(board, chess.engine.Limit(nodes=nodes), root_moves=[mv], game=(fen, uci))
    s = info["score"].pov(board.turn)
    return fen, uci, s.score(mate_score=100_000), s.mate()


def eval_moves(pairs: list[tuple[str, str]], sf_path: str, nodes: int = NODES, workers: int | None = None,
               progress: bool = True) -> dict[tuple[str, str], tuple[int, int | None]]:
    """{(fen, uci): (cp, mate)} for each requested (fen, uci)."""
    jobs = [(f, u, nodes) for f, u in dict.fromkeys(pairs)]
    res = {}
    if not jobs:
        return res
    it = None
    with Pool(workers or os.cpu_count() or 4, initializer=_init, initargs=(sf_path,)) as pool:
        it = pool.imap_unordered(_eval_move, jobs, chunksize=8)
        if progress:
            from tqdm.auto import tqdm
            it = tqdm(it, total=len(jobs), desc="sf eval moves")
        for fen, uci, cp, mate in it:
            res[(fen, uci)] = (cp, mate)
    return res
