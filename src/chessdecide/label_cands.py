"""Attach Stockfish win-prob to each candidate and pick the gold candidate (lowest wp_loss)."""
from __future__ import annotations

import json

import pandas as pd

from .engine import win_prob


def label(pos: pd.DataFrame, cdf: pd.DataFrame, extra: dict) -> pd.DataFrame:
    """pos: positions (id, fen, sf_multipv, sf_best_uci, sf_best_wp); cdf: id, cands (json).
    extra: {(fen, uci): (cp, mate)} for candidates outside MultiPV.
    Returns cdf + cands (with wp, wp_loss), gold_uci, gold_san, best_in_cands, gold_wp_loss."""
    p = pos.set_index("id")
    out = []
    for r in cdf.itertuples(index=False):
        row = p.loc[r.id]
        mpv = {d["uci"]: win_prob(d["cp"], d["mate"]) for d in json.loads(row["sf_multipv"])}
        cands = json.loads(r.cands)
        for c in cands:
            if c["uci"] in mpv:
                w = mpv[c["uci"]]
            else:
                ev = extra.get((row["fen"], c["uci"]))
                w = win_prob(*ev) if ev else 0.0
            c["wp"] = w
            c["wp_loss"] = max(0.0, row["sf_best_wp"] - w)
        # ties broken by LLM order (stable)
        gold = min(cands, key=lambda c: c["wp_loss"])
        d = r._asdict()
        d.update(cands=json.dumps(cands), gold_uci=gold["uci"], gold_san=gold["san"],
                 gold_wp_loss=gold["wp_loss"], best_in_cands=row["sf_best_uci"] in {c["uci"] for c in cands})
        out.append(d)
    return pd.DataFrame(out)
