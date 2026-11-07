# %% [markdown]
# # 02 — Raw LLM (Qwen3-8B): pilot metrics + candidate generation
#
# **Kaggle:** Accelerator **GPU T4 ×2** · Internet **On** ·
# Inputs: notebook output `chess-01-data`, dataset `chessdecide-lib`.
#
# * `MODE = "pilot"`: 500 pilot positions (≈30 min). Measures C1, C2, oracle (X2), random (X3),
#   first-candidate, X1-base, and candidate recall@k. **Go/no-go gate for the whole experiment.**
# * `MODE = "gen"`: same, plus LLM₀ candidates for every split (train/val/test/test_ood),
#   needed to train D\* (C4). Several hours; resumable by chunk.

# %%
import glob, json, os, subprocess, sys, time

def sh(cmd):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)

MODE = "pilot"
LLM_ID = "Qwen/Qwen3-8B"
K = 5
SEED = 0
RUN_X1 = True          # base GLiNER over all legal moves (pilot only)
GEN_BUDGET_H = 9.5

def find(pattern):
    hits = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    assert hits, f"input not found: {pattern} (did you attach the inputs?)"
    return hits[0]

POS = find("positions.parquet")
LIB = os.path.dirname(os.path.dirname(find("chessdecide/__init__.py")))
sys.path.insert(0, LIB)
os.environ["PYTHONPATH"] = LIB + ":" + os.environ.get("PYTHONPATH", "")
WORK = "/kaggle/working"
GEN_DIR = f"{WORK}/gen_llm0"
print(POS, LIB, sep="\n")

# %%
sh(f"{sys.executable} -m pip install -q vllm chess")
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
sh(f"{sys.executable} -m pip list 2>/dev/null | grep -iE '^(vllm|torch|transformers) '")

# %%
import numpy as np
import pandas as pd
import chess

pos = pd.read_parquet(POS)
pilot = pos[pos["pilot"]]
if MODE == "pilot":
    sel = pilot
else:
    prio = {"val": 1, "test": 2, "test_ood": 3, "train": 4}
    sel = pd.concat([pilot, pos[~pos["pilot"]].assign(_p=lambda d: d["split"].map(prio)).sort_values("_p")])
ids_file = f"{WORK}/gen_ids.txt"
open(ids_file, "w").write("\n".join(sel["id"]))
open(f"{WORK}/pilot_ids.txt", "w").write("\n".join(pilot["id"]))
print(MODE, len(sel), "positions;", sel["split"].value_counts().to_dict())

# %% [markdown]
# ## Generate (subprocess → GPU fully freed afterwards)

# %%
t0 = time.time()
sh(f"{sys.executable} -m chessdecide.gen_llm --positions {POS} --ids {ids_file} --out {GEN_DIR} "
   f"--model {LLM_ID} --k {K} --single-splits val,test,test_ood --time-budget-h {GEN_BUDGET_H}")
gen_minutes = (time.time() - t0) / 60

# %%
from chessdecide.proposer import parse_candidates, parse_single

gen = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f"{GEN_DIR}/part_*.parquet"))])
g = gen.merge(pos[["id", "fen"]], on="id")
rows = []
for r in g.itertuples(index=False):
    b = chess.Board(r.fen)
    mv = parse_single(b, r.single_raw) if isinstance(r.single_raw, str) else None
    cands, n_valid, n_pad = parse_candidates(b, r.cands_raw or "", K)
    rows.append(dict(id=r.id, split=r.split, single_uci=mv.uci() if mv else None,
                     cands=json.dumps(cands), n_valid=n_valid, n_pad=n_pad,
                     single_raw=r.single_raw, cands_raw=r.cands_raw))
cdf = pd.DataFrame(rows)
cdf.to_parquet(f"{WORK}/llm0_cands.parquet", index=False)
print(f"{len(cdf):,} parsed; mean valid LLM candidates {cdf.n_valid.mean():.2f}/{K}; "
      f"positions needing padding {(cdf.n_pad > 0).mean():.1%}")
print("sample raw:", cdf["cands_raw"].iloc[0][:400])

# %% [markdown]
# ## Base GLiNER decider on the pilot (C2) and over all legal moves (X1-base)

# %%
sh(f'{sys.executable} -m pip install -q "gliner2[local]"')
sh(f"{sys.executable} -m chessdecide.run_decider --positions {POS} --cands {WORK}/llm0_cands.parquet "
   f"--ids {WORK}/pilot_ids.txt --out {WORK}/pilot_c2.parquet")
if RUN_X1:
    sh(f"{sys.executable} -m chessdecide.run_decider --positions {POS} --all-legal "
       f"--ids {WORK}/pilot_ids.txt --out {WORK}/pilot_x1base.parquet")

# %% [markdown]
# ## Pilot metrics

# %%
from chessdecide import engine, metrics

p = pilot.merge(cdf.drop(columns=["split"]), on="id")
p["cand_ucis"] = p["cands"].map(lambda s: [c["uci"] for c in json.loads(s)])
p["llm_ucis"] = p["cands"].map(lambda s: [c["uci"] for c in json.loads(s) if not c["padded"]])
rng = np.random.default_rng(SEED)
p["c1"] = p["single_uci"]
p["first"] = p["cand_ucis"].map(lambda c: c[0])
p["rand"] = p["cand_ucis"].map(lambda c: c[rng.integers(len(c))])
p = p.merge(pd.read_parquet(f"{WORK}/pilot_c2.parquet")[["id", "chosen_uci"]].rename(columns={"chosen_uci": "c2"}), on="id")
if RUN_X1:
    p = p.merge(pd.read_parquet(f"{WORK}/pilot_x1base.parquet")[["id", "chosen_uci"]].rename(columns={"chosen_uci": "x1b"}), on="id")
move_cols = ["c1", "cand_ucis", "c2"] + (["x1b"] if RUN_X1 else [])

sf = engine.get_stockfish()
extra = engine.eval_moves(metrics.needed_evals(p, move_cols), sf)
p["oracle"] = [max(cs, key=lambda u: metrics.move_wp(r, u, extra)) for cs, (_, r) in zip(p["cand_ucis"], p.iterrows())]

names = {"C1 LLM0 single": "c1", "LLM0 first cand": "first", "X3 random cand": "rand",
         "C2 LLM0→D0": "c2", "X2 oracle cand": "oracle"}
if RUN_X1:
    names["X1-base D0 all legal"] = "x1b"
scored = {n: metrics.score_moves(p, c, extra) for n, c in names.items()}
table = metrics.summarize(scored)
recall = {
    "recall@k (best in final cands)": float(np.mean([b in c for b, c in zip(p.sf_best_uci, p.cand_ucis)])),
    "recall@k (best in LLM-only cands)": float(np.mean([b in c for b, c in zip(p.sf_best_uci, p.llm_ucis)])),
    "C1 legal rate": float(p.c1.notna().mean()),
    "mean valid LLM cands": float(p.n_valid.mean()),
}
by_source = pd.DataFrame({n: s["top1"].groupby(p["source"]).mean() for n, s in scored.items()}).T
pvals = {"C2 vs C1": metrics.mcnemar(scored["C2 LLM0→D0"]["top1"], scored["C1 LLM0 single"]["top1"])}

pd.set_option("display.width", 200)
print(table[["n", "legal", "top1", "top1_ci", "top3", "wp_loss", "blunder"]])
print(json.dumps(recall, indent=1))
print("top-1 by source:\n", by_source.round(3))
print("McNemar:", pvals)

gap = table.loc["X2 oracle cand", "top1"] - table.loc["C1 LLM0 single", "top1"]
verdict = ("GO: oracle beats C1 by %.1f pts top-1, so a better picker has room to help." % (100 * gap)
           if gap >= 0.05 else
           "CAUTION: oracle ≈ C1 (%.1f pts). Candidate list is the bottleneck; raise k / improve prompt." % (100 * gap))
print(verdict)

report = dict(mode=MODE, llm=LLM_ID, k=K, n_pilot=len(p), gen_minutes=round(gen_minutes, 1),
              table=json.loads(table.to_json(orient="index")), recall=recall,
              top1_by_source=json.loads(by_source.to_json()), mcnemar=pvals, verdict=verdict)
json.dump(report, open(f"{WORK}/pilot_report.json", "w"), indent=2)
p.drop(columns=["single_raw", "cands_raw"]).to_parquet(f"{WORK}/pilot_rows.parquet", index=False)
print(sorted(os.listdir(WORK)))
