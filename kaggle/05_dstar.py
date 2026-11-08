# %% [markdown]
# # 05 — Fine-tune the decision model (D\*, the "cheap" arm): GLiNER2.5-Decide
#
# **Kaggle:** Accelerator **GPU T4 ×2** (uses one) · Internet **On** ·
# Inputs: `chess-01-data`, `chess-04-label-cands`, dataset `chessdecide-lib`.
#
# Text = rendered position, labels = LLM₀'s k candidates (with the LLM's rationale as description),
# gold = the Stockfish-best candidate. Same train positions as LLM\*.
# Trains the data-scaling curve (5k / 10k / 20k); the largest model is D\* for notebook 06.
# Output: `dstar_<n>/` checkpoints, `dstar_val.json`, `cost_dstar.json`.

# %%
import glob, inspect, json, os, shutil, subprocess, sys, time

def sh(cmd):
    print("$", cmd, flush=True); subprocess.run(cmd, shell=True, check=True)

def find(pattern):
    hits = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    assert hits, f"input not found: {pattern}"
    return hits[0]

QUICK = True
BASE = "fastino/GLiNER2.5-Decide"
SIZES = [20_000, 5_000, 10_000]   # largest first: it is D* for notebook 06
EPOCHS = 3
BATCH = 8
ENC_LR, TASK_LR = 1e-5, 5e-4
SEED = 0
if QUICK:
    SIZES, EPOCHS = [100], 3

sh(f'{sys.executable} -m pip install -q chess "gliner2[local]"')
POS = find("positions.parquet")
LAB = find("llm0_cands_labeled.parquet")
_pkg = os.path.dirname(find("proposer.py"))
LIB = "/kaggle/tmp/lib"; os.makedirs(LIB, exist_ok=True)
if not os.path.exists(f"{LIB}/chessdecide"):
    os.symlink(_pkg, f"{LIB}/chessdecide")
sys.path.insert(0, LIB)
os.environ["PYTHONPATH"] = LIB + ":" + os.environ.get("PYTHONPATH", "")
WORK = "/kaggle/working"

# %%
import chess
import numpy as np
import pandas as pd
import torch
from chessdecide.decider import GlinerDecider
pos = pd.read_parquet(POS)
lab = pd.read_parquet(LAB).merge(pos[["id", "fen", "source", "sf_best_uci"]], on="id")
train_all = lab[lab["split"] == "train"].sample(frac=1.0, random_state=SEED)
val = lab[lab["split"] == "val"]
if QUICK and len(train_all) == 0:  # smoke test on pilot-only candidates: train on val just to exercise the API
    train_all = val.copy()
print(f"train {len(train_all):,} / val {len(val):,}; best-in-cands train {train_all.best_in_cands.mean():.1%}")


def evaluate(model_path, df):
    dec = GlinerDecider(model_path, device="cuda")
    picks = dec.choose_batch([chess.Board(f) for f in df["fen"]], [json.loads(c) for c in df["cands"]])
    chosen = [u for u, _ in picks]
    wpl = [next(c["wp_loss"] for c in json.loads(cs) if c["uci"] == u) for u, cs in zip(chosen, df["cands"])]
    del dec; torch.cuda.empty_cache()
    return dict(top1=float(np.mean(np.array(chosen) == df["sf_best_uci"].values)),
                gold_acc=float(np.mean(np.array(chosen) == df["gold_uci"].values)),
                wp_loss=float(np.mean(wpl)), blunder=float(np.mean(np.array(wpl) > 0.2)))


# baselines on val: base decider (C2-val), first candidate, oracle
results = {"D0 (base)": evaluate(BASE, val)}
results["first cand"] = dict(top1=float(np.mean([json.loads(c)[0]["uci"] == b for c, b in zip(val.cands, val.sf_best_uci)])))
results["oracle"] = dict(top1=float(val.best_in_cands.mean()), wp_loss=float(val.gold_wp_loss.mean()))
print(json.dumps(results, indent=1))

# %%
# Each size trains in its own process (fresh AMP/optimizer state); fp16 first, fp32 retry if it was a no-op.
costs = []
for n in SIZES:
    n = min(n, len(train_all))
    out = f"{WORK}/dstar_{n}"
    for prec in ["fp16", "fp32"]:
        rc = subprocess.run(f"{sys.executable} -m chessdecide.train_decider --labeled {LAB} --positions {POS} "
                            f"--n {n} --out {out} --epochs {EPOCHS} --batch {BATCH} --encoder-lr {ENC_LR} "
                            f"--task-lr {TASK_LR} --precision {prec} --seed {SEED}", shell=True).returncode
        if rc == 0:
            break
        print(f"n={n}: {prec} run failed (rc={rc})" + ("; retrying in fp32" if prec == "fp16" else ""), flush=True)
    if rc != 0:
        results[f"D* n={n}"] = dict(error=f"training failed rc={rc}")
        continue
    meta = json.load(open(f"{out}/train_meta.json"))
    r = evaluate(out, val)
    results[f"D* n={n}"] = r
    costs.append(dict(arm="D*", **meta, gpu_hours=round(meta["train_seconds"] / 3600, 3), val=r))
    print(f"n={n}: {meta['train_seconds']/60:.1f} min ({meta['precision']}), val {r}", flush=True)
    if n != max(SIZES):  # keep only the largest checkpoint in the output (it is D* for notebook 06)
        shutil.rmtree(out, ignore_errors=True)
    # save after every size so partial results survive a crash
    json.dump(results, open(f"{WORK}/dstar_val.json", "w"), indent=2)
    json.dump(dict(runs=costs, quick=QUICK, note="LLM0 candidate generation cost for D* training is in notebook 02 gen_minutes"),
              open(f"{WORK}/cost_dstar.json", "w"), indent=2)

print(pd.DataFrame(results).T.round(4))
print(sorted(os.listdir(WORK)))
