"""Batch-generate LLM outputs (single best move and/or k candidates) for a set of positions.

Run as a subprocess so the GPU is fully released afterwards:
  python -m chessdecide.gen_llm --positions positions.parquet --ids ids.txt --out gen/ \
      --single-splits val,test,test_ood [--lora path/to/adapter] [--backend auto|vllm|hf]

Writes gen/part_XXXX.parquet with: id, split, single_raw, cands_raw, prompt_version, model, lora.
Resumable: existing parts are skipped (by id).
"""
from __future__ import annotations

import argparse
import glob
import os
import time

import chess
import pandas as pd

from .proposer import PROMPT_VERSION, candidates_messages, single_move_messages

MAX_TOK = {"single": 48, "cands": 256}


def build_prompts(tok, df, k, single_splits):
    jobs = []  # (id, kind, prompt)
    for r in df.itertuples(index=False):
        b = chess.Board(r.fen)
        msgs = {"cands": candidates_messages(b, k)}
        if r.split in single_splits:
            msgs["single"] = single_move_messages(b)
        for kind, m in msgs.items():
            p = tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            jobs.append((r.id, kind, p))
    return jobs


class VLLMBackend:
    def __init__(self, model, tp, lora, max_len):
        from vllm import LLM
        self.llm = LLM(model=model, dtype="half", tensor_parallel_size=tp, max_model_len=max_len,
                       gpu_memory_utilization=0.90, seed=0, enable_lora=bool(lora), max_lora_rank=64,
                       enable_prefix_caching=True)
        self.lora = None
        if lora:
            from vllm.lora.request import LoRARequest
            self.lora = LoRARequest("ft", 1, lora)

    def generate(self, prompts, max_toks):
        from vllm import SamplingParams
        sps = [SamplingParams(temperature=0.0, max_tokens=m, seed=0) for m in max_toks]
        outs = self.llm.generate(prompts, sps, lora_request=self.lora, use_tqdm=True)
        return [o.outputs[0].text for o in outs]


class HFBackend:
    def __init__(self, model, lora, bs=16):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float16, device_map="auto")
        if lora:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, lora)
        self.model.eval()
        self.bs = bs

    def generate(self, prompts, max_toks):
        from tqdm.auto import tqdm
        out = [None] * len(prompts)
        # group by max_tokens so a batch shares one limit
        for mt in sorted(set(max_toks)):
            idx = [i for i, m in enumerate(max_toks) if m == mt]
            idx.sort(key=lambda i: len(prompts[i]))
            for s in tqdm(range(0, len(idx), self.bs), desc=f"hf gen (max {mt})"):
                b = idx[s:s + self.bs]
                enc = self.tok([prompts[i] for i in b], return_tensors="pt", padding=True).to(self.model.device)
                with self.torch.no_grad():
                    g = self.model.generate(**enc, max_new_tokens=mt, do_sample=False,
                                            pad_token_id=self.tok.pad_token_id)
                texts = self.tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
                for i, t in zip(b, texts):
                    out[i] = t
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", required=True)
    ap.add_argument("--ids", help="file with one id per line (default: all)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--lora")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--single-splits", default="val,test,test_ood")
    ap.add_argument("--backend", default="auto", choices=["auto", "vllm", "hf"])
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--time-budget-h", type=float, default=10.0)
    a = ap.parse_args()

    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    df = pd.read_parquet(a.positions)
    if a.ids:
        order = [l.strip() for l in open(a.ids) if l.strip()]
        df = df.set_index("id").loc[order].reset_index()
    done = set()
    for p in glob.glob(os.path.join(a.out, "part_*.parquet")):
        done |= set(pd.read_parquet(p, columns=["id"])["id"])
    df = df[~df["id"].isin(done)].reset_index(drop=True)
    print(f"{len(df):,} positions to generate ({len(done):,} already done)", flush=True)
    if df.empty:
        return

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    backend = None
    if a.backend in ("auto", "vllm"):
        try:
            backend = VLLMBackend(a.model, a.tp, a.lora, a.max_len)
            print("backend: vllm", flush=True)
        except Exception as e:  # noqa: BLE001
            if a.backend == "vllm":
                raise
            print(f"vLLM unavailable ({e!r}); falling back to HF transformers", flush=True)
    if backend is None:
        backend = HFBackend(a.model, a.lora)
        print("backend: hf", flush=True)

    singles = set(s for s in a.single_splits.split(",") if s)
    part = len(glob.glob(os.path.join(a.out, "part_*.parquet")))
    for s in range(0, len(df), a.chunk):
        if (time.time() - t0) / 3600 > a.time_budget_h:
            print("time budget reached, stopping (resume later)", flush=True)
            break
        sub = df.iloc[s:s + a.chunk]
        jobs = build_prompts(tok, sub, a.k, singles)
        t1 = time.time()
        texts = backend.generate([j[2] for j in jobs], [MAX_TOK[j[1]] for j in jobs])
        res = {i: {"id": i} for i in sub["id"]}
        for (i, kind, _), t in zip(jobs, texts):
            res[i][f"{kind}_raw"] = t
        out = pd.DataFrame(list(res.values())).merge(sub[["id", "split"]], on="id")
        for c in ["single_raw", "cands_raw"]:
            if c not in out:
                out[c] = None
        out["prompt_version"], out["model"], out["lora"] = PROMPT_VERSION, a.model, a.lora or ""
        out.to_parquet(os.path.join(a.out, f"part_{part:04d}.parquet"), index=False)
        part += 1
        print(f"chunk {s // a.chunk}: {len(sub)} positions, {len(jobs)} prompts in {time.time() - t1:.0f}s "
              f"(total {(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
