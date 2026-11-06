#!/usr/bin/env python3
'Token-clean TwoSeq-same mean-versus-last-token extraction.'

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from contextlib import contextmanager

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
TOX_DIR = os.path.join(ROOT, "toxicity")
sys.path.insert(0, TOX_DIR)
import run_all_toxicity_eval as ev

QUOTE_SYSTEM = (
    "You are a communication style consultant. "
    "Your job is to help rephrase messages in different tones while preserving the core meaning."
)


def polite_messages(text):
    return [
        {"role": "system", "content": QUOTE_SYSTEM},
        {
            "role": "user",
            "content": (
                f'A user wrote this message:\n"{text}"\n\n'
                "Now rephrase the same core message in a professional, polite, and appropriate way "
                "suitable for a public or workplace setting. The goal is to express the same idea "
                "but in a respectful tone:"
            ),
        },
    ]


def apply_chat(tokenizer, messages):
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def stop_ids(tokenizer):
    ids = {tokenizer.eos_token_id}
    for token in ("<|eot_id|>", "<|im_end|>"):
        tid = tokenizer.convert_tokens_to_ids(token)
        if tid is not None and tid != tokenizer.unk_token_id:
            ids.add(tid)
    return ids


@contextmanager
def capture_all_layers(model):
    layers = ev.get_layers(model)
    captured = [None] * len(layers)
    handles = []

    def make_hook(index):
        def hook(_module, _inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            captured[index] = h.detach().float().cpu()

        return hook

    for index, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_hook(index)))
    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()


def generate_all_layers(model, tokenizer, messages, max_new_tokens):
    prompt = apply_chat(tokenizer, messages)
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    stops = stop_ids(tokenizer)
    with torch.no_grad():
        out = model(ids, use_cache=True, return_dict=True)
    past = out.past_key_values
    next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated_ids = []
    generated_resids = []

    for _ in range(max_new_tokens):
        token_id = int(next_id.item())
        with capture_all_layers(model) as cap:
            with torch.no_grad():
                out = model(
                    next_id,
                    use_cache=True,
                    past_key_values=past,
                    return_dict=True,
                )
        if any(x is None for x in cap):
            raise RuntimeError("missing layer capture")
        generated_resids.append(
            torch.stack([x[0, -1].clone() for x in cap], dim=0)
        )
        generated_ids.append(token_id)
        past = out.past_key_values
        if token_id in stops:
            break
        next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    lexical = [
        (tid, resid)
        for tid, resid in zip(generated_ids, generated_resids)
        if tid not in stops
    ]
    if not lexical:
        raise RuntimeError("empty lexical continuation")
    lex_ids = [x[0] for x in lexical]
    lex_resids = torch.stack([x[1] for x in lexical], dim=0)
    return {
        "text": tokenizer.decode(lex_ids, skip_special_tokens=True).strip(),
        "mean_resid": lex_resids.mean(dim=0),
        "last_resid": lex_resids[-1],
    }


def normalize_layers(vec):
    out = vec.float().clone()
    return out / out.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def aggregate_pair_diffs(diffs):
    normalized = [normalize_layers(x) for x in diffs]
    return normalize_layers(torch.stack(normalized).mean(dim=0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alias", default="Llama-3.1-8B")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=150)
    args = parser.parse_args()

    cfg = next(x for x in ev.MODELS_CONFIG if x["alias"] == args.alias)
    tokenizer = AutoTokenizer.from_pretrained(cfg["path"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"], torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to("cuda:0")
    model.eval()

    cache_path = os.path.join(
        ev.OUTPUT_DIR,
        f"paradetox_splits_seed{ev.PD_SEED}_train{ev.N_PD_TRAIN_CANDS}_eval{ev.N_PD_EVAL}.json",
    )
    train_pairs, _ = ev.load_paradetox_splits(
        n_train_cands=ev.N_PD_TRAIN_CANDS,
        n_eval=ev.N_PD_EVAL,
        seed=ev.PD_SEED,
        cache_path=cache_path if os.path.exists(cache_path) else None,
    )
    random.Random(args.seed).shuffle(train_pairs)
    chosen = train_pairs[: args.n]

    mean_diffs = []
    last_diffs = []
    records = []
    refusals = {"arm1": 0, "arm2": 0}
    for i, (toxic, _neutral) in enumerate(chosen):
        arm1 = generate_all_layers(
            model, tokenizer, ev._residual_chat_messages_stageA(toxic), args.max_new_tokens
        )
        arm2 = generate_all_layers(
            model, tokenizer, polite_messages(toxic), args.max_new_tokens
        )
        refusals["arm1"] += int(ev._is_refusal(arm1["text"]))
        refusals["arm2"] += int(ev._is_refusal(arm2["text"]))
        mean_diffs.append(arm2["mean_resid"] - arm1["mean_resid"])
        last_diffs.append(arm2["last_resid"] - arm1["last_resid"])
        records.append(
            {
                "pair_index": i,
                "toxic": toxic,
                "arm1": arm1["text"],
                "arm2": arm2["text"],
            }
        )
        print(f"[pair] seed={args.seed} {i + 1}/{args.n}", flush=True)

    vectors = {
        "avg": aggregate_pair_diffs(mean_diffs),
        "last": aggregate_pair_diffs(last_diffs),
    }
    alias_dir = os.path.join(ev.VECTOR_DIR, args.alias.replace(".", "_"))
    os.makedirs(alias_dir, exist_ok=True)
    saved = {}
    for readout, vector in vectors.items():
        suffix = f"strict_twoseq_{readout}_s{args.seed}_n{args.n}"
        path = os.path.join(alias_dir, f"vector_paradetox_{suffix}.pt")
        torch.save(vector, path)
        saved[readout] = {"suffix": suffix, "path": path}
        print(path, flush=True)

    out_dir = os.path.join(HERE, "out_strict_twoseq")
    os.makedirs(out_dir, exist_ok=True)
    with open(
        os.path.join(out_dir, f"extract_seed{args.seed}_n{args.n}.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "seed": args.seed,
                "n": args.n,
                "refusals": refusals,
                "saved": saved,
                "records": records,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    main()
