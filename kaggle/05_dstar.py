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
import glob, inspect, json, os, subprocess, sys, time

def sh(cmd):
    print("$", cmd, flush=True); subprocess.run(cmd, shell=True, check=True)

def find(pattern):
    hits = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    assert hits, f"input not found: {pattern}"
    return hits[0]

QUICK = True
BASE = "fastino/GLiNER2.5-Decide"
SIZES = [5_000, 10_000, 20_000]
EPOCHS = 3
BATCH = 8
ENC_LR, TASK_LR = 1e-5, 5e-4
SEED = 0
if QUICK:
    SIZES, EPOCHS = [100], 1

sh(f'{sys.executable} -m pip install -q chess "gliner2[local]"')
POS = find("positions.parquet")
LAB = find("llm0_cands_labeled.parquet")
_pkg = os.path.dirname(find("proposer.py"))
LIB = "/kaggle/tmp/lib"; os.makedirs(LIB, exist_ok=True)
if not os.path.exists(f"{LIB}/chessdecide"):
    os.symlink(_pkg, f"{LIB}/chessdecide")
sys.path.insert(0, LIB)
WORK = "/kaggle/working"

# %%
import chess
import numpy as np
import pandas as pd
import torch
from gliner2 import AutoExtractor
from gliner2.training.data import Classification, InputExample, TrainingDataset
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig

from chessdecide.decider import TASK, GlinerDecider, render_labels, render_text

print("TrainingConfig params:", list(inspect.signature(TrainingConfig).parameters))
pos = pd.read_parquet(POS)
lab = pd.read_parquet(LAB).merge(pos[["id", "fen", "source", "sf_best_uci"]], on="id")
train_all = lab[lab["split"] == "train"].sample(frac=1.0, random_state=SEED)
val = lab[lab["split"] == "val"]
print(f"train {len(train_all):,} / val {len(val):,}; best-in-cands train {train_all.best_in_cands.mean():.1%}")


def to_examples(df):
    exs = []
    for r in df.itertuples(index=False):
        labs = render_labels(json.loads(r.cands))
        exs.append(InputExample(text=render_text(chess.Board(r.fen)), classifications=[
            Classification(task=TASK, labels=list(labs), true_label=r.gold_san, label_descriptions=labs)]))
    return exs


val_ex = to_examples(val.head(500))


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
costs = []
for n in SIZES:
    n = min(n, len(train_all))
    out = f"{WORK}/dstar_{n}"
    model = AutoExtractor.from_pretrained(BASE).to("cuda")
    want = dict(output_dir=f"/kaggle/tmp/dstar_ckpt_{n}", experiment_name=f"dstar_{n}", num_epochs=EPOCHS,
                batch_size=BATCH, eval_batch_size=16, encoder_lr=ENC_LR, task_lr=TASK_LR, seed=SEED,
                fp16=True, bf16=False, logging_steps=50, report_to="none")
    params = inspect.signature(TrainingConfig).parameters
    cfg = TrainingConfig(**{k: v for k, v in want.items() if k in params})
    trainer = GLiNER2Trainer(model=model, config=cfg, train_data=TrainingDataset(to_examples(train_all.head(n))),
                             eval_data=TrainingDataset(val_ex))
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    secs = time.time() - t0
    model.save_pretrained(out)
    del trainer, model; torch.cuda.empty_cache()
    r = evaluate(out, val)
    results[f"D* n={n}"] = r
    costs.append(dict(arm="D*", n_train_positions=n, epochs=EPOCHS, gpu=torch.cuda.get_device_name(0), n_gpus_used=1,
                      trainable_params=486_400_000, train_hours=round(secs / 3600, 3), gpu_hours=round(secs / 3600, 3),
                      peak_vram_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2), val=r))
    print(f"n={n}: {secs/60:.1f} min, val {r}", flush=True)

print(pd.DataFrame(results).T.round(4))
json.dump(results, open(f"{WORK}/dstar_val.json", "w"), indent=2)
json.dump(dict(runs=costs, quick=QUICK, note="LLM0 candidate generation cost for D* training is in notebook 02 gen_minutes"),
          open(f"{WORK}/cost_dstar.json", "w"), indent=2)
print(sorted(os.listdir(WORK)))
