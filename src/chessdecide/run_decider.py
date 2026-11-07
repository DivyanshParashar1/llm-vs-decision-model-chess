"""Run a GLiNER decider over candidate lists (or all legal moves) as a subprocess.

  python -m chessdecide.run_decider --positions positions.parquet --cands cands.parquet \
      --out picks.parquet [--model fastino/GLiNER2.5-Decide | path/to/D*] [--all-legal]

cands.parquet: id, cands (json list of {uci, san, why, padded}).
Writes: id, chosen_uci, scores (json), decider, mode.
"""
from __future__ import annotations

import argparse
import json
import time

import chess
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", required=True)
    ap.add_argument("--cands", help="parquet with id, cands (json); omit with --all-legal")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="fastino/GLiNER2.5-Decide")
    ap.add_argument("--all-legal", action="store_true", help="labels = every legal move (control X1)")
    ap.add_argument("--no-descriptions", action="store_true")
    ap.add_argument("--ids")
    a = ap.parse_args()

    import torch
    from tqdm.auto import tqdm

    from .decider import GlinerDecider

    pos = pd.read_parquet(a.positions).set_index("id")
    if a.all_legal:
        ids = [l.strip() for l in open(a.ids)] if a.ids else list(pos.index)
        cand_map = {}
        for i in ids:
            b = chess.Board(pos.at[i, "fen"])
            cand_map[i] = [{"uci": m.uci(), "san": b.san(m), "why": ""} for m in b.legal_moves]
    else:
        cdf = pd.read_parquet(a.cands)
        if a.ids:
            keep = set(l.strip() for l in open(a.ids))
            cdf = cdf[cdf["id"].isin(keep)]
        cand_map = {r.id: json.loads(r.cands) for r in cdf.itertuples(index=False)}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dec = GlinerDecider(a.model, device=device, use_descriptions=not a.no_descriptions)
    t0, rows = time.time(), []
    for i, cands in tqdm(cand_map.items(), desc=f"decider ({device})"):
        b = chess.Board(pos.at[i, "fen"])
        (uci, scores), = dec.choose_batch([b], [cands])
        rows.append(dict(id=i, chosen_uci=uci, scores=json.dumps(scores)))
    out = pd.DataFrame(rows)
    out["decider"], out["mode"] = a.model, "all_legal" if a.all_legal else "cands"
    out.to_parquet(a.out, index=False)
    print(f"{len(out)} picks in {time.time() - t0:.0f}s on {device}; "
          f"ms/pos={(time.time() - t0) * 1000 / max(1, len(out)):.0f}")


if __name__ == "__main__":
    main()
