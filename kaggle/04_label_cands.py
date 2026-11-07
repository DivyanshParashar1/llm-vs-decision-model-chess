# %% [markdown]
# # 04 — Stockfish-label the raw LLM's candidates (CPU only, no GPU quota)
#
# **Kaggle:** Accelerator **None** · Internet **On** ·
# Inputs: `chess-01-data`, `chess-02-llm0` (run in `MODE="gen"`), dataset `chessdecide-lib`.
#
# For every candidate outside the stored MultiPV, evaluate it with Stockfish (same settings as 01),
# then gold = candidate with the lowest win-prob loss. Also caches evals of C1's single moves.
# Output: `llm0_cands_labeled.parquet`, `sf_evals.parquet` (eval cache: fen, uci, cp, mate).

# %%
import glob, json, os, subprocess, sys, time

def sh(cmd):
    print("$", cmd, flush=True); subprocess.run(cmd, shell=True, check=True)

def find(pattern):
    hits = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    assert hits, f"input not found: {pattern}"
    return hits[0]

sh(f"{sys.executable} -m pip install -q chess")
POS = find("positions.parquet")
CANDS = find("llm0_cands.parquet")
_pkg = os.path.dirname(find("proposer.py"))
LIB = "/kaggle/tmp/lib"; os.makedirs(LIB, exist_ok=True)
if not os.path.exists(f"{LIB}/chessdecide"):
    os.symlink(_pkg, f"{LIB}/chessdecide")
sys.path.insert(0, LIB)
WORK = "/kaggle/working"

# %%
import pandas as pd
from chessdecide import engine, metrics
from chessdecide.label_cands import label

pos = pd.read_parquet(POS)
cdf = pd.read_parquet(CANDS)
print(f"{len(cdf):,} positions with candidates; splits {cdf['split'].value_counts().to_dict()}")
p = pos.merge(cdf[["id", "cands", "single_uci"]], on="id")
p["cand_ucis"] = p["cands"].map(lambda s: [c["uci"] for c in json.loads(s)])
pairs = metrics.needed_evals(p, ["cand_ucis", "single_uci"])
print(f"{len(pairs):,} extra Stockfish evals needed")

t0 = time.time()
sf = engine.get_stockfish()
extra = engine.eval_moves(pairs, sf)
minutes = (time.time() - t0) / 60
print(f"evals done in {minutes:.1f} min on {os.cpu_count()} CPUs")

pd.DataFrame([dict(fen=f, uci=u, cp=cp, mate=m) for (f, u), (cp, m) in extra.items()]) \
  .to_parquet(f"{WORK}/sf_evals.parquet", index=False)
lab = label(pos, cdf, extra)
lab.to_parquet(f"{WORK}/llm0_cands_labeled.parquet", index=False)

print(lab.groupby("split")[["best_in_cands", "gold_wp_loss"]].mean().round(3))
print("gold position in LLM order:",
      lab.apply(lambda r: [c["uci"] for c in json.loads(r.cands)].index(r.gold_uci), axis=1)
         .value_counts(normalize=True).sort_index().round(3).to_dict())
json.dump(dict(n=len(lab), extra_evals=len(pairs), minutes=round(minutes, 1)),
          open(f"{WORK}/label_meta.json", "w"), indent=2)
