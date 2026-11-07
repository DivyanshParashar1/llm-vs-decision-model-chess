import json, sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from chessdecide import metrics as M
from chessdecide.engine import win_prob

MPV = json.dumps([{"uci": "h5f7", "cp": 100000, "mate": 1}, {"uci": "g1f3", "cp": 50, "mate": None},
                  {"uci": "d2d3", "cp": 20, "mate": None}])
DF = pd.DataFrame({"fen": ["F"] * 4, "sf_multipv": [MPV] * 4, "sf_best_uci": ["h5f7"] * 4,
                   "sf_best_wp": [1.0] * 4, "move": ["h5f7", "d2d3", "a2a3", None]})


def test_needed_evals_only_outside_multipv():
    assert M.needed_evals(DF, ["move"]) == [("F", "a2a3")]


def test_score_moves():
    s = M.score_moves(DF, "move", {("F", "a2a3"): (-300, None)})
    assert s["legal"].tolist() == [True, True, True, False]
    assert s["top1"].tolist() == [True, False, False, False]
    assert s["top3"].tolist() == [True, True, False, False]
    assert s["wp_loss"].iloc[0] == 0.0
    assert abs(s["wp_loss"].iloc[1] - (1 - win_prob(20))) < 1e-9
    assert s["wp_loss"].iloc[3] == 1.0 and s["blunder"].iloc[3]


def test_summary_and_mcnemar():
    s = M.score_moves(DF, "move", {("F", "a2a3"): (-300, None)})
    t = M.summarize({"X": s}, n_boot=100)
    assert t.loc["X", "top1"] == 0.25
    assert M.mcnemar([True] * 10, [False] * 10) < 0.01
    assert M.mcnemar([True, False], [True, False]) == 1.0
