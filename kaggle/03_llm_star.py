# %% [markdown]
# # 03 — Fine-tune the LLM (LLM\*, the "expensive" arm): QLoRA on Qwen3-8B with Unsloth
#
# **Kaggle:** Accelerator **GPU T4 ×2** (Unsloth uses one) · Internet **On** ·
# Inputs: notebook output `chess-01-data`, dataset `chessdecide-lib`.
# Can run on a **second Kaggle account** in parallel with notebook 02 (needs only the data from 01).
#
# Each train position → one example, 50/50 single-move or k-candidate format (targets = Stockfish),
# so the same LLM\* serves C3 (alone) and C5 (as proposer). Loss on assistant tokens only.
# Output: `llm_star_adapter/` (LoRA) + `cost_llm_star.json` (GPU-hours etc. for the cost comparison).

# %%
import glob, json, os, random, subprocess, sys, time

def sh(cmd):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)

QUICK = True
BASE = "unsloth/Qwen3-8B-unsloth-bnb-4bit"   # 4-bit Qwen3-8B; the adapter is applied to Qwen/Qwen3-8B at eval
N_TRAIN = 20_000          # train positions used (scaling curve: 5k / 10k / 20k)
K = 5
SEED = 0
EPOCHS = 1
LR = 1e-4
BATCH, ACCUM = 8, 2
MAX_LEN = 1024
TIME_BUDGET_H = 10.0      # stop + save before Kaggle's session limit
N_VAL_CHECK = 200
if QUICK:
    N_TRAIN, N_VAL_CHECK = 160, 40

def find(pattern):
    hits = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    assert hits, f"input not found: {pattern}"
    return hits[0]

POS = find("positions.parquet")
_pkg = os.path.dirname(find("proposer.py"))
LIB = "/kaggle/tmp/lib"; os.makedirs(LIB, exist_ok=True)
if not os.path.exists(f"{LIB}/chessdecide"):
    os.symlink(_pkg, f"{LIB}/chessdecide")
sys.path.insert(0, LIB)
WORK = "/kaggle/working"
OUT = f"{WORK}/llm_star_adapter"

# %%
sh(f"{sys.executable} -m pip install -q unsloth chess")

# %%
import pandas as pd
import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from datasets import Dataset
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer

from chessdecide.sft import make_example

pos = pd.read_parquet(POS)
train = pos[pos["split"] == "train"].sample(frac=1.0, random_state=SEED).head(N_TRAIN)
val = pos[pos["split"] == "val"].sample(frac=1.0, random_state=SEED).head(N_VAL_CHECK)
print(len(train), "train positions;", train["source"].value_counts().to_dict())

t_load = time.time()
model, tok = FastLanguageModel.from_pretrained(BASE, max_seq_length=MAX_LEN, load_in_4bit=True, dtype=None)
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth", random_state=SEED)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"trainable {trainable/1e6:.1f}M / total {total/1e6:.0f}M (4-bit counted packed)")

rng = random.Random(SEED)
texts, fmts = [], []
for r in train.itertuples(index=False):
    msgs, target, fmt = make_example(r.fen, r.sf_multipv, K, rng)
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    texts.append(prompt + target + "<|im_end|>\n"); fmts.append(fmt)
n_tokens = sum(len(tok(t)["input_ids"]) for t in texts[:500]) / min(500, len(texts)) * len(texts)
print(f"{len(texts)} examples, ~{n_tokens/1e6:.1f}M tokens, formats {pd.Series(fmts).value_counts().to_dict()}")
print(texts[0][-400:])
ds = Dataset.from_dict({"text": texts})

# %%
class TimeBudget(TrainerCallback):
    def __init__(self, hours): self.t0, self.limit = time.time(), hours * 3600
    def on_step_end(self, args, state, control, **kw):
        if time.time() - self.t0 > self.limit:
            print("time budget hit: stopping early"); control.should_training_stop = True

cfg = SFTConfig(
    output_dir=f"/kaggle/tmp/llm_star_ckpt", dataset_text_field="text", max_seq_length=MAX_LEN,
    per_device_train_batch_size=BATCH, gradient_accumulation_steps=ACCUM, num_train_epochs=EPOCHS,
    max_steps=20 if QUICK else -1, learning_rate=LR, lr_scheduler_type="cosine", warmup_steps=5 if QUICK else 40,
    logging_steps=10, fp16=not torch.cuda.is_bf16_supported(), bf16=torch.cuda.is_bf16_supported(),
    optim="adamw_8bit", weight_decay=0.0, seed=SEED, save_strategy="no", report_to="none", packing=False)
try:
    trainer = SFTTrainer(model=model, processing_class=tok, train_dataset=ds, args=cfg, callbacks=[TimeBudget(TIME_BUDGET_H)])
except TypeError:
    trainer = SFTTrainer(model=model, tokenizer=tok, train_dataset=ds, args=cfg, callbacks=[TimeBudget(TIME_BUDGET_H)])
trainer = train_on_responses_only(trainer, instruction_part="<|im_start|>user\n", response_part="<|im_start|>assistant\n")

torch.cuda.reset_peak_memory_stats()
t0 = time.time()
stats = trainer.train()
train_s = time.time() - t0
model.save_pretrained(OUT); tok.save_pretrained(OUT)
print(f"trained {train_s/3600:.2f} h, final loss {stats.training_loss:.4f} → {OUT}")

# %% [markdown]
# ## Quick sanity check on val (single-move top-1, HF generation). Full eval happens in notebook 05.

# %%
import chess
from chessdecide.proposer import parse_single, single_move_messages

FastLanguageModel.for_inference(model)
tok.padding_side = "left"
correct = legal = 0
for s in range(0, len(val), 16):
    sub = val.iloc[s:s + 16]
    prompts = [tok.apply_chat_template(single_move_messages(chess.Board(f)), tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False) for f in sub["fen"]]
    enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        g = model.generate(**enc, max_new_tokens=48, do_sample=False)
    outs = tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    for (_, r), o in zip(sub.iterrows(), outs):
        mv = parse_single(chess.Board(r.fen), o)
        legal += mv is not None
        correct += mv is not None and mv.uci() == r.sf_best_uci
print(f"val sanity (n={len(val)}): legal {legal/len(val):.1%}, top-1 {correct/len(val):.1%}; sample: {outs[0]!r}")

cost = dict(arm="LLM*", base=BASE, method="QLoRA r16 a32 (Unsloth)", gpu=torch.cuda.get_device_name(0),
            n_gpus_used=1, trainable_params=trainable, n_train_positions=len(train), n_examples=len(texts),
            approx_train_tokens=int(n_tokens), epochs=EPOCHS, steps=trainer.state.global_step,
            train_hours=round(train_s / 3600, 3), gpu_hours=round(train_s / 3600, 3),
            peak_vram_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2), final_loss=stats.training_loss,
            val_sanity_top1=correct / len(val), val_sanity_legal=legal / len(val), quick=QUICK)
json.dump(cost, open(f"{WORK}/cost_llm_star.json", "w"), indent=2)
print(json.dumps(cost, indent=2))
