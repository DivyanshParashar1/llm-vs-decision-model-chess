import json, sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from chessdecide.label_cands import label

MPV = json.dumps([{"uci": "h5f7", "cp": 100000, "mate": 1}, {"uci": "g1f3", "cp": 50, "mate": None}])
POS = pd.DataFrame({"id": ["a"], "fen": ["F"], "sf_multipv": [MPV], "sf_best_uci": ["h5f7"], "sf_best_wp": [1.0]})


def test_gold_is_lowest_loss_and_extra_evals_used():
    cands = [{"uci": "d2d3", "san": "d3"}, {"uci": "g1f3", "san": "Nf3"}, {"uci": "a2a3", "san": "a3"}]
    cdf = pd.DataFrame({"id": ["a"], "cands": [json.dumps(cands)]})
    out = label(POS, cdf, {("F", "d2d3"): (30, None), ("F", "a2a3"): (-200, None)})
    r = out.iloc[0]
    assert r.gold_uci == "g1f3" and not r.best_in_cands
    wps = [c["wp"] for c in json.loads(r.cands)]
    assert wps[2] < wps[0] < wps[1]
