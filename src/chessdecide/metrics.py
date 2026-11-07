"""Per-position scoring + aggregate metrics with bootstrap CIs and McNemar."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .engine import win_prob

BLUNDER_WP = 0.20


def move_wp(row, uci: str | None, extra_evals: dict) -> float | None:
    """wp of playing `uci` in row's position: MultiPV line if present, else an extra SF eval."""
    if uci is None:
        return None
    for d in json.loads(row["sf_multipv"]):
        if d["uci"] == uci:
            return win_prob(d["cp"], d["mate"])
    ev = extra_evals.get((row["fen"], uci))
    return None if ev is None else win_prob(*ev)


def needed_evals(df: pd.DataFrame, move_cols: list[str]) -> list[tuple[str, str]]:
    """(fen, uci) pairs whose wp is not covered by the stored MultiPV."""
    out = []
    for _, r in df.iterrows():
        mpv = {d["uci"] for d in json.loads(r["sf_multipv"])}
        for c in move_cols:
            v = r[c]
            for u in (v if isinstance(v, (list, np.ndarray)) else [v]):
                if isinstance(u, str) and u not in mpv:
                    out.append((r["fen"], u))
    return out


def score_moves(df: pd.DataFrame, col: str, extra_evals: dict) -> pd.DataFrame:
    """Adds legal / top1 / top3 / wp_loss / blunder for the move in `col` (uci or None)."""
    top3 = df["sf_multipv"].map(lambda s: [d["uci"] for d in json.loads(s)[:3]])
    out = pd.DataFrame(index=df.index)
    out["legal"] = df[col].map(lambda u: isinstance(u, str))
    out["top1"] = df[col] == df["sf_best_uci"]
    out["top3"] = [isinstance(u, str) and u in t for u, t in zip(df[col], top3)]
    wps = [move_wp(r, r[col] if isinstance(r[col], str) else None, extra_evals) for _, r in df.iterrows()]
    # illegal/unparsed = worst case (wp 0)
    out["wp_loss"] = [max(0.0, b - (w if w is not None else 0.0)) for b, w in zip(df["sf_best_wp"], wps)]
    out["blunder"] = out["wp_loss"] > BLUNDER_WP
    return out


def bootstrap_ci(x, n=10_000, seed=0):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    return (x.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5))


def summarize(scored: dict[str, pd.DataFrame], n_boot=2_000) -> pd.DataFrame:
    rows = []
    for name, s in scored.items():
        r = {"config": name, "n": len(s)}
        for m in ["legal", "top1", "top3", "wp_loss", "blunder"]:
            mean, lo, hi = bootstrap_ci(s[m], n=n_boot)
            r[m] = round(mean, 4)
            r[f"{m}_ci"] = f"[{lo:.3f}, {hi:.3f}]"
        rows.append(r)
    return pd.DataFrame(rows).set_index("config")


def mcnemar(a, b) -> float:
    """Exact two-sided McNemar p-value on paired booleans."""
    from scipy.stats import binomtest
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    if n01 + n10 == 0:
        return 1.0
    return binomtest(n01, n01 + n10, 0.5).pvalue
