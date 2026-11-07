# %% [markdown]
# # 06 — Final evaluation: C1–C5 + controls on the frozen test set (and test_ood)
#
# **Kaggle:** Accelerator **GPU T4 ×2** · Internet **On** · Inputs: `chess-01-data`, `chess-02-llm0`,
# `chess-03-llm-star`, `chess-04-label-cands`, `chess-05-dstar`, dataset `chessdecide-lib`.
#
# | ID | config | source of moves |
# |----|--------|-----------------|
# | C1 | LLM₀ alone | notebook 02 single-move output |
# | C2 | LLM₀ → D₀ | base GLiNER over LLM₀ candidates |
# | C3 | LLM\* alone | LLM\* single-move (generated here) |
# | C4 | LLM₀ → D\* | fine-tuned GLiNER over LLM₀ candidates |
# | C5 | LLM\* → D\* | fine-tuned GLiNER over LLM\* candidates (generated here) |
# | X1 | D\* over all legal moves | no LLM |
# | X2 | oracle over LLM₀ candidates | upper bound for C2/C4 |
# | X3 | random LLM₀ candidate | lower bound |

# %%
import glob, json, os, re, subprocess, sys, time

def sh(cmd):
    print("$", cmd, flush=True); subprocess.run(cmd, shell=True, check=True)

def find(pattern):
    hits = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    assert hits, f"input not found: {pattern}"
    return hits[0]

LLM_ID = "Qwen/Qwen3-8B"
K = 5
SEED = 0
SPLITS = ["test", "test_ood"]
QUICK = False

POS = find("positions.parquet")
CANDS0 = find("llm0_cands.parquet")
ADAPTER = os.path.dirname(find("llm_star_adapter/adapter_config.json"))
dstars = glob.glob("/kaggle/input/**/dstar_*/", recursive=True)
DSTAR = max(dstars, key=lambda p: int(re.search(r"dstar_(\d+)", p).group(1))).rstrip("/")
SF_CACHE = find("sf_evals.parquet")
_pkg = os.path.dirname(find("proposer.py"))
LIB = "/kaggle/tmp/lib"; os.makedirs(LIB, exist_ok=True)
if not os.path.exists(f"{LIB}/chessdecide"):
    os.symlink(_pkg, f"{LIB}/chessdecide")
sys.path.insert(0, LIB)
os.environ["PYTHONPATH"] = LIB + ":" + os.environ.get("PYTHONPATH", "")
WORK = "/kaggle/working"
print(POS, CANDS0, ADAPTER, DSTAR, sep="\n")

# %%
sh(f"{sys.executable} -m pip install -q vllm==0.33.0 chess")
import numpy as np
import pandas as pd
import chess

pos = pd.read_parquet(POS)
ev = pos[pos["split"].isin(SPLITS)]
if QUICK:
    ev = ev.groupby("split").head(30)
ids_file = f"{WORK}/eval_ids.txt"
open(ids_file, "w").write("\n".join(ev["id"]))
print(ev["split"].value_counts().to_dict())

# %% [markdown]
# ## LLM\* generation (single move for C3, candidates for C5)

# %%
t0 = time.time()
sh(f"{sys.executable} -m chessdecide.gen_llm --positions {POS} --ids {ids_file} --out {WORK}/gen_llmstar "
   f"--model {LLM_ID} --lora {ADAPTER} --k {K} --single-splits {','.join(SPLITS)}")
llmstar_gen_s = time.time() - t0

from chessdecide.proposer import parse_candidates, parse_single

def parse_dir(d):
    g = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f"{d}/part_*.parquet"))]).merge(pos[["id", "fen"]], on="id")
    rows = []
    for r in g.itertuples(index=False):
        b = chess.Board(r.fen)
        mv = parse_single(b, r.single_raw) if isinstance(r.single_raw, str) else None
        cands, nv, npad = parse_candidates(b, r.cands_raw or "", K)
        rows.append(dict(id=r.id, single_uci=mv.uci() if mv else None, cands=json.dumps(cands), n_valid=nv, n_pad=npad))
    return pd.DataFrame(rows)

cs = parse_dir(f"{WORK}/gen_llmstar")
cs.to_parquet(f"{WORK}/llmstar_cands.parquet", index=False)
c0 = pd.read_parquet(CANDS0)
c0 = c0[c0["id"].isin(ev["id"])][["id", "single_uci", "cands", "n_valid", "n_pad"]]
c0.to_parquet(f"{WORK}/llm0_cands_eval.parquet", index=False)

# %% [markdown]
# ## Deciders

# %%
sh(f'{sys.executable} -m pip install -q "gliner2[local]"')
timing = {}
def run_dec(name, model, cands=None, all_legal=False):
    t = time.time()
    extra = "--all-legal" if all_legal else f"--cands {cands}"
    sh(f"{sys.executable} -m chessdecide.run_decider --positions {POS} {extra} --ids {ids_file} "
       f"--model {model} --out {WORK}/picks_{name}.parquet")
    timing[name] = (time.time() - t) / len(ev) * 1000
    return pd.read_parquet(f"{WORK}/picks_{name}.parquet")[["id", "chosen_uci"]].rename(columns={"chosen_uci": name})

picks = [run_dec("c2", "fastino/GLiNER2.5-Decide", f"{WORK}/llm0_cands_eval.parquet"),
         run_dec("c4", DSTAR, f"{WORK}/llm0_cands_eval.parquet"),
         run_dec("c5", DSTAR, f"{WORK}/llmstar_cands.parquet"),
         run_dec("x1", DSTAR, all_legal=True)]

# %% [markdown]
# ## Score everything

# %%
from chessdecide import engine, metrics

p = ev.merge(c0.rename(columns={"single_uci": "c1", "cands": "cands0", "n_valid": "nv0"}), on="id") \
      .merge(cs.rename(columns={"single_uci": "c3", "cands": "cands_s", "n_valid": "nvs"})[["id", "c3", "cands_s", "nvs"]], on="id")
for d in picks:
    p = p.merge(d, on="id")
p["u0"] = p["cands0"].map(lambda s: [c["uci"] for c in json.loads(s)])
p["us"] = p["cands_s"].map(lambda s: [c["uci"] for c in json.loads(s)])
rng = np.random.default_rng(SEED)
p["x3"] = p["u0"].map(lambda c: c[rng.integers(len(c))])

cache = pd.read_parquet(SF_CACHE)
extra = {(r.fen, r.uci): (r.cp, None if pd.isna(r.mate) else int(r.mate)) for r in cache.itertuples(index=False)}
need = [x for x in metrics.needed_evals(p, ["c1", "c2", "c3", "c4", "c5", "x1", "u0", "us"]) if x not in extra]
print(f"{len(need):,} new Stockfish evals")
extra.update(engine.eval_moves(need, engine.get_stockfish()))
p["x2"] = [max(u, key=lambda m: metrics.move_wp(r, m, extra)) for u, (_, r) in zip(p["u0"], p.iterrows())]

NAMES = {"C1 LLM0": "c1", "C2 LLM0→D0": "c2", "C3 LLM*": "c3", "C4 LLM0→D*": "c4", "C5 LLM*→D*": "c5",
         "X1 D* all-legal": "x1", "X2 oracle(LLM0 cands)": "x2", "X3 random(LLM0 cands)": "x3"}
results = {}
for split in SPLITS:
    q = p[p["split"] == split].reset_index(drop=True)
    sc = {n: metrics.score_moves(q, c, extra) for n, c in NAMES.items()}
    tab = metrics.summarize(sc, n_boot=10_000)
    tab["recall@k"] = np.nan
    tab.loc["C2 LLM0→D0", "recall@k"] = tab.loc["C4 LLM0→D*", "recall@k"] = np.mean([b in u for b, u in zip(q.sf_best_uci, q.u0)])
    tab.loc["C5 LLM*→D*", "recall@k"] = np.mean([b in u for b, u in zip(q.sf_best_uci, q.us)])
    pairs = [("C4 LLM0→D*", "C3 LLM*"), ("C4 LLM0→D*", "C1 LLM0"), ("C5 LLM*→D*", "C3 LLM*"),
             ("C5 LLM*→D*", "C4 LLM0→D*"), ("C2 LLM0→D0", "C1 LLM0"), ("X1 D* all-legal", "C4 LLM0→D*")]
    mc = {f"{a} vs {b}": metrics.mcnemar(sc[a]["top1"], sc[b]["top1"]) for a, b in pairs}
    slices = {}
    for col in ["source", "phase", "band"]:
        slices[col] = pd.DataFrame({n: s["top1"].groupby(q[col]).mean() for n, s in sc.items()}).T.round(3)
    # full-line puzzle solve: every included solver step correct (Lichess move, SF best, or mates)
    def ok(r, u):
        if not isinstance(u, str):
            return False
        if u in (r.ref_uci, r.sf_best_uci):
            return True
        b = chess.Board(r.fen); b.push_uci(u)
        return b.is_checkmate()
    pz = q[q.source == "puzzle"]
    full = {n: float(pz.assign(_ok=[ok(r, getattr(r, c)) for r in pz.itertuples()]).groupby("group")["_ok"].all().mean())
            for n, c in NAMES.items()} if len(pz) else {}
    results[split] = dict(table=tab, mcnemar=mc, slices=slices, full_line_puzzle_solve=full, n=len(q))
    pd.set_option("display.width", 220)
    print(f"\n===== {split} (n={len(q)}) =====")
    print(tab[["legal", "top1", "top1_ci", "top3", "wp_loss", "wp_loss_ci", "blunder", "recall@k"]])
    print("McNemar p:", json.dumps({k: round(v, 4) for k, v in mc.items()}, indent=1))
    print("full-line puzzle solve:", {k: round(v, 3) for k, v in full.items()})
    for col, s in slices.items():
        print(f"top-1 by {col}:\n{s}")

# %% [markdown]
# ## Cost table + save

# %%
def maybe_json(pattern):
    try:
        return json.load(open(find(pattern)))
    except AssertionError:
        return None

cost = dict(llm_star=maybe_json("cost_llm_star.json"), dstar=maybe_json("cost_dstar.json"),
            llm0_generation=maybe_json("pilot_report.json"),
            latency_ms_per_move=dict(decider=timing, llmstar_gen_s_total=round(llmstar_gen_s, 1)))
print(json.dumps(cost, indent=1, default=str)[:3000])

out = {s: dict(n=r["n"], table=json.loads(r["table"].to_json(orient="index")), mcnemar=r["mcnemar"],
               full_line_puzzle_solve=r["full_line_puzzle_solve"],
               slices={k: json.loads(v.to_json()) for k, v in r["slices"].items()}) for s, r in results.items()}
out["cost"] = cost
json.dump(out, open(f"{WORK}/results.json", "w"), indent=2, default=str)
for s, r in results.items():
    r["table"].to_csv(f"{WORK}/table_{s}.csv")
p.drop(columns=["cands0", "cands_s"]).to_parquet(f"{WORK}/eval_rows.parquet", index=False)
print(sorted(os.listdir(WORK)))
