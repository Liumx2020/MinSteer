#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import json
import argparse
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "run_all_toxicity_eval", "toxicity_eval_results")
N_PD_EVAL = 100
PD_SEED = 42


def load_paradetox_eval(n_eval: int = N_PD_EVAL, seed: int = PD_SEED):
    cache_path = os.path.join(OUTPUT_DIR, f"paradetox_splits_seed{seed}_train1000_eval{n_eval}.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
        return cached.get("eval_samples", [])
    from datasets import load_dataset
    import random
    ds = load_dataset("s-nlp/paradetox", split="train")
    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    eval_idxs = idxs[1000 : 1000 + n_eval]
    return [
        {"toxic": str(ds[i]["en_toxic_comment"]).strip(), "neutral_reference": str(ds[i].get("en_neutral_comment", "")).strip()}
        for i in eval_idxs if str(ds[i]["en_toxic_comment"]).strip()
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_n", type=int, default=N_PD_EVAL)
    ap.add_argument("--seed", type=int, default=PD_SEED)
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    print("加载 BART-base-detox...")
    model_path = "s-nlp/bart-base-detox"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.eval()
    if __import__("torch").cuda.is_available():
        model = model.cuda()

    samples = load_paradetox_eval(n_eval=args.eval_n, seed=args.seed)
    print(f"Eval 样本数: {len(samples)}")

    results = []
    for ex in tqdm(samples, desc="BART detoxicfy"):
        toxic = ex["toxic"]
        inputs = tokenizer(toxic, return_tensors="pt", truncation=True, max_length=512)
        if model.device.type == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with __import__("torch").no_grad():
            out = model.generate(**inputs, max_length=512, num_beams=4)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        results.append({
            "toxic_text": toxic,
            "neutral_reference": ex.get("neutral_reference"),
            "baseline_or_intervention_output": text,
        })

    out_path = os.path.join(OUTPUT_DIR, "BART-base-detox_paradetox_baseline.json")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model": "BART-base-detox",
            "path": model_path,
            "label": "paradetox_baseline",
            "results": results,
            "n_prompts": len(results),
        }, f, indent=2, ensure_ascii=False)
    print(f"已保存 -> {out_path}")
    print("可运行 python eval_paradetox_metrics.py 计算 STA/SIM/FL")


if __name__ == "__main__":
    main()
