"""Unit tests for the pure helpers in kaggle/01_data.py (no network, no Stockfish).

Run:  CHESS_SKIP_INSTALL=1 pytest tests/
"""
import io
import os
import pathlib

import chess
import numpy as np
import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "kaggle" / "01_data.py"
STOP = "# %% [markdown]\n# ## GLiNER2.5-Decide smoke test"


@pytest.fixture(scope="module")
def m(tmp_path_factory):
    os.environ["CHESS_SKIP_INSTALL"] = "1"
    cwd = os.getcwd()
    os.chdir(tmp_path_factory.mktemp("work"))
    ns = {"__name__": "nb01"}
    exec(compile(SRC.read_text().split(STOP)[0], str(SRC), "exec"), ns)
    os.chdir(cwd)
    return ns


def test_puzzle_fen_gotcha(m):
    # real Lichess puzzle 00sHx: FEN is before the opponent's move e8d7
    recs = m["explode_puzzle"]("00sHx", "q3k1nr/1pp1nQpp/3p4/1P2p3/4P3/B1PP1b2/B5PP/5K2 b k - 0 17",
                               "e8d7 a2e6 d7d8 f7f8", cap=3)
    assert [r["ref_san"] for r in recs] == ["Be6+", "Qf8#"]
    assert all(chess.Board(r["fen"]).turn == chess.WHITE for r in recs)
    assert [r["step"] for r in recs] == [0, 1] and recs[0]["n_steps"] == 2


def test_puzzle_cap(m):
    recs = m["explode_puzzle"]("x", "q3k1nr/1pp1nQpp/3p4/1P2p3/4P3/B1PP1b2/B5PP/5K2 b k - 0 17",
                               "e8d7 a2e6 d7d8 f7f8", cap=1)
    assert len(recs) == 1


def test_illegal_puzzle_raises(m):
    with pytest.raises(ValueError):
        m["explode_puzzle"]("x", chess.STARTING_FEN, "e2e5 e7e5", cap=3)


def test_phase_and_wp(m):
    assert m["phase_of"](chess.Board()) == "opening"
    assert m["phase_of"](chess.Board("8/5k2/8/8/8/8/3R4/4K3 w - - 0 60")) == "endgame"
    assert m["win_prob"](0) == pytest.approx(0.5)
    assert m["win_prob"](0, mate=3) == 1.0 and m["win_prob"](0, mate=-2) == 0.0
    assert m["norm_fen"](chess.STARTING_FEN) == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"


PGN = """[Event "Rated Blitz game"]
[Site "https://lichess.org/abcdefgh"]
[WhiteElo "1500"]
[BlackElo "1600"]
[TimeControl "180+0"]
[Termination "Normal"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Nb8 10. d4 Nbd7 11. Nbd2 Bb7 12. Bc2 Re8 13. Nf1 Bf8 14. Ng3 g6 1-0

[Event "Rated Bullet game"]
[Site "https://lichess.org/zzzzzzzz"]
[WhiteElo "1500"]
[BlackElo "1600"]
[TimeControl "60+0"]
[Termination "Normal"]

1. e4 e5 1-0
"""


def test_pgn_split_and_filter(m):
    texts = list(m["iter_game_texts"](io.StringIO(PGN)))
    assert len(texts) == 2
    hs = [m["parse_headers"](t) for t in texts]
    assert hs[0]["Site"].endswith("abcdefgh")
    assert m["game_ok"](hs[0], 180) and not m["game_ok"](hs[1], 180)


def test_game_positions(m):
    t = next(m["iter_game_texts"](io.StringIO(PGN)))
    h = m["parse_headers"](t)
    recs = m["sample_game_positions"](t, h, "abcdefgh", 8, 3, np.random.default_rng(0))
    assert 1 <= len(recs) <= 3
    for r in recs:
        b = chess.Board(r["fen"])
        assert chess.Move.from_uci(r["ref_uci"]) in b.legal_moves
        assert int(r["id"].split("_")[-1]) >= 8


def test_assign_splits_grouped(m):
    import pandas as pd
    df = pd.DataFrame({"group": [f"g{i // 2}" for i in range(40)]})
    s = m["assign_splits"](df, {"test": 6, "val": 6, "train": 20}, seed=0)
    assert (df.assign(s=s).groupby("group")["s"].nunique() == 1).all()
    assert (s == "test").sum() == 6 and (s == "train").sum() == 20
