#!/usr/bin/env python3
"""Token-aligned sentiment transcript controls."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import random
import re
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
CLF_ID = "cardiffnlp/twitter-roberta-base-sentiment"
TEXT_FILE = ROOT / "sentiment" / "data" / "test_text.txt"
LABEL_FILE = ROOT / "sentiment" / "data" / "test_labels.txt"
OUT = HERE / "out"
CONDITIONS = ("self_transcript", "quoted_self", "quoted_mismatch", "twoseq")


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_texts():
    texts = [
        x.rstrip("\n").strip('"')
        for x in TEXT_FILE.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    labels = [
        int(x.strip())
        for x in LABEL_FILE.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    if len(texts) != len(labels):
        raise RuntimeError(f"text/label mismatch: {len(texts)} vs {len(labels)}")
    return texts, labels


def get_layers(model):
    return model.model.layers


def stop_ids(tokenizer):
    values = {tokenizer.eos_token_id}
    for token in ("<|eot_id|>", "<|im_end|>"):
        tid = tokenizer.convert_tokens_to_ids(token)
        if tid is not None and tid != tokenizer.unk_token_id:
            values.add(tid)
    values.discard(None)
    return values


def render(tokenizer, messages):
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def negative_messages(text: str):
    return [
        {
            "role": "system",
            "content": "You rewrite social-media messages while preserving their core meaning.",
        },
        {
            "role": "user",
            "content": (
                "Rewrite the following tweet in a negative tone while preserving its "
                f"meaning:\n\n{text}"
            ),
        },
    ]


def positive_messages(text: str):
    return [
        {
            "role": "system",
            "content": "You rewrite social-media messages while preserving their core meaning.",
        },
        {
            "role": "user",
            "content": (
                "Rewrite the following tweet in a positive tone while preserving its "
                f"meaning:\n\n{text}"
            ),
        },
    ]


def conversion_message():
    return {
        "role": "user",
        "content": (
            "Now rewrite the same message in a positive tone. "
            "Keep the same core meaning."
        ),
    }


def quoted_messages(source_text: str, stage_a_text: str):
    return [
        {
            "role": "system",
            "content": "You rewrite social-media messages while preserving their core meaning.",
        },
        {
            "role": "user",
            "content": (
                f"Original source tweet:\n{source_text}\n\n"
                "A negative-tone rewrite associated with that source is:\n"
                f"{stage_a_text}\n\n"
                "Rewrite the associated negative-tone message in a positive tone. "
                "Preserve the original source tweet core meaning."
            ),
        },
    ]


@contextlib.contextmanager
def capture_all_layers(model):
    captured = [None] * len(get_layers(model))
    handles = []

    def make_hook(index):
        def hook(_module, _inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            captured[index] = h.detach().float().cpu()
        return hook

    for index, layer in enumerate(get_layers(model)):
        handles.append(layer.register_forward_hook(make_hook(index)))
    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()


def forward_capture(model, ids, past=None):
    with capture_all_layers(model) as captured:
        with torch.no_grad():
            out = model(
                ids,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
    if any(x is None for x in captured):
        raise RuntimeError("missing residual capture")
    residuals = torch.stack(captured, dim=0)  # [L, 1, T, H]
    return out.logits.detach().float().cpu(), residuals, out.past_key_values


def clean_generate(model, tokenizer, messages, max_new_tokens):
    """Greedy generation whose residual row is aligned to the forwarded token."""
    prompt = render(tokenizer, messages)
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        out = model(prompt_ids, use_cache=True, return_dict=True)
    past = out.past_key_values
    next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    stops = stop_ids(tokenizer)
    lexical_ids = []
    lexical_residuals = []
    stop_id = None

    for _ in range(max_new_tokens):
        token_id = int(next_id.item())
        if token_id in stops:
            stop_id = token_id
            break
        _, residual, past = forward_capture(model, next_id, past)
        lexical_ids.append(token_id)
        lexical_residuals.append(residual[:, 0, 0].clone())
        with torch.no_grad():

            # keeps this helper simple and the token ledger explicit.
            out = model(next_id, past_key_values=None, use_cache=False, return_dict=True)
        # Do not use the recomputed logits: obtain the correct cached next-token logits
        # in one extra lightweight cached forward is both wasteful and wrong. The


        raise RuntimeError("unreachable ledger guard")

    if not lexical_ids:
        raise RuntimeError("empty lexical generation")
    return {
        "messages": messages,
        "prompt": prompt,
        "prompt_ids": prompt_ids[0].detach().cpu().tolist(),
        "lexical_ids": lexical_ids,
        "text": tokenizer.decode(
            lexical_ids, skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip(),
        "residuals": torch.stack(lexical_residuals),
        "stop_id": stop_id,
    }


def clean_generate_fixed(model, tokenizer, messages, max_new_tokens):
    """Correct implementation kept separate so the ledger is easy to audit."""
    prompt = render(tokenizer, messages)
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        out = model(prompt_ids, use_cache=True, return_dict=True)
    past = out.past_key_values
    next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    stops = stop_ids(tokenizer)
    lexical_ids = []
    lexical_residuals = []
    stop_id = None

    for _ in range(max_new_tokens):
        token_id = int(next_id.item())
        if token_id in stops:
            stop_id = token_id
            break
        logits, residual, past = forward_capture(model, next_id, past)
        lexical_ids.append(token_id)
        lexical_residuals.append(residual[:, 0, 0].clone())
        next_id = logits[:, -1, :].to(model.device).argmax(dim=-1, keepdim=True)

    if not lexical_ids:
        raise RuntimeError("empty lexical generation")
    residuals = torch.stack(lexical_residuals)  # [T,L,H]
    if residuals.shape[0] != len(lexical_ids):
        raise AssertionError("token/residual count mismatch")
    return {
        "messages": messages,
        "prompt": prompt,
        "prompt_ids": prompt_ids[0].detach().cpu().tolist(),
        "lexical_ids": lexical_ids,
        "text": tokenizer.decode(
            lexical_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip(),
        "residuals": residuals,
        "mean_resid": residuals.mean(dim=0),
        "stop_id": stop_id,
    }


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
    ).to("cuda:0")
    model.eval()
    return model, tokenizer


def selected_pool(seed: int, n: int):
    texts, _ = load_texts()
    pool = texts[500:]
    indices = random.Random(seed).sample(range(len(pool)), n)
    return pool, indices, [pool[i] for i in indices]


def aggregate_raw(diffs):
    return torch.stack(diffs).mean(dim=0)


def extract_condition(args):
    model, tokenizer = load_model()
    pool, indices, chosen = selected_pool(args.seed, args.n)

    stage_as = [
        clean_generate_fixed(
            model, tokenizer, negative_messages(text), args.max_new_tokens
        )
        for text in chosen
    ]
    diffs = []
    records = []
    for i, (pool_index, source, stage_a) in enumerate(
        zip(indices, chosen, stage_as)
    ):
        if args.condition == "self_transcript":
            messages_b = (
                negative_messages(source)
                + [{"role": "assistant", "content": stage_a["text"]}]
                + [conversion_message()]
            )
        elif args.condition == "quoted_self":
            messages_b = quoted_messages(source, stage_a["text"])
        elif args.condition == "quoted_mismatch":
            mismatch = stage_as[(i + 1) % len(stage_as)]
            messages_b = quoted_messages(source, mismatch["text"])
        elif args.condition == "twoseq":
            messages_b = positive_messages(source)
        else:
            raise ValueError(args.condition)

        stage_b = clean_generate_fixed(
            model, tokenizer, messages_b, args.max_new_tokens
        )
        diff = stage_b["mean_resid"] - stage_a["mean_resid"]
        diffs.append(diff)
        records.append(
            {
                "pool_index": pool_index,
                "source": source,
                "stage_a_text": stage_a["text"],
                "stage_a_lexical_ids": stage_a["lexical_ids"],
                "stage_a_stop_id": stage_a["stop_id"],
                "stage_b_text": stage_b["text"],
                "stage_b_lexical_ids": stage_b["lexical_ids"],
                "stage_b_stop_id": stage_b["stop_id"],
                "stage_a_token_count": len(stage_a["lexical_ids"]),
                "stage_b_token_count": len(stage_b["lexical_ids"]),
                "raw_diff_layer_norms": diff.norm(dim=-1).tolist(),
            }
        )
        print(
            f"[{args.condition}] {i + 1}/{args.n} "
            f"A={len(stage_a['lexical_ids'])} B={len(stage_b['lexical_ids'])}",
            flush=True,
        )

    vector = aggregate_raw(diffs)
    run_dir = OUT / f"extract_{args.condition}_seed{args.seed}_n{args.n}"
    run_dir.mkdir(parents=True, exist_ok=True)
    vector_path = run_dir / "vector.pt"
    torch.save(vector, vector_path)
    atomic_json(
        run_dir / "manifest.json",
        {
            "task": "TweetEval sentiment negative-to-positive",
            "model": MODEL_ID,
            "dtype": "bfloat16",
            "condition": args.condition,
            "seed": args.seed,
            "n": args.n,
            "pool": "TweetEval test rows 500 onward",
            "pool_indices": indices,
            "max_new_tokens": args.max_new_tokens,
            "readout": "mean resid_post over generated lexical tokens only",
            "aggregation": "raw mean over extraction units",
            "vector_shape": list(vector.shape),
            "vector_layer_norms": vector.norm(dim=-1).tolist(),
            "vector_sha256": sha256(vector_path),
            "records": records,
        },
    )
    print(vector_path, flush=True)


def cache_check(args):
    model, tokenizer = load_model()
    _, indices, chosen = selected_pool(args.seed, args.n)
    rows = []

    for i, (pool_index, source) in enumerate(zip(indices, chosen)):
        messages_a = negative_messages(source)
        stage_a = clean_generate_fixed(
            model, tokenizer, messages_a, args.max_new_tokens
        )
        messages_b = (
            messages_a
            + [{"role": "assistant", "content": stage_a["text"]}]
            + [conversion_message()]
        )
        stage_b = clean_generate_fixed(
            model, tokenizer, messages_b, args.max_new_tokens
        )

        ids_a = torch.tensor([stage_a["prompt_ids"]], device=model.device)
        ids_a_lex = torch.tensor([stage_a["lexical_ids"]], device=model.device)
        full_prompt_ids = torch.tensor(
            [stage_b["prompt_ids"]], device=model.device
        )
        direct_prefix = torch.cat([ids_a, ids_a_lex], dim=1)
        prefix = full_prompt_ids[:, : direct_prefix.shape[1]]
        prefix_exact = torch.equal(prefix, direct_prefix)
        if not prefix_exact:
            raise RuntimeError(
                f"canonical prefix mismatch for pool_index={pool_index}"
            )
        suffix = full_prompt_ids[:, direct_prefix.shape[1] :]
        ids_b_lex = torch.tensor(
            [stage_b["lexical_ids"]], device=model.device
        )

        # Incremental cached execution.
        _, _, past = forward_capture(model, ids_a, None)
        _, _, past = forward_capture(model, ids_a_lex, past)
        _, _, past = forward_capture(model, suffix, past)
        logits_c, resid_c, _ = forward_capture(model, ids_b_lex, past)


        ledger = torch.cat([full_prompt_ids, ids_b_lex], dim=1)
        logits_r_full, resid_r_full, _ = forward_capture(model, ledger, None)
        t = ids_b_lex.shape[1]
        logits_r = logits_r_full[:, -t:]
        resid_r = resid_r_full[:, :, -t:]

        resid_cos = []
        for layer in range(resid_c.shape[0]):
            resid_cos.append(
                F.cosine_similarity(
                    resid_c[layer, 0], resid_r[layer, 0], dim=-1
                ).mean().item()
            )
        crd_c = resid_c[:, 0].mean(dim=1) - stage_a["mean_resid"]
        crd_r = resid_r[:, 0].mean(dim=1) - stage_a["mean_resid"]
        crd_cos = F.cosine_similarity(crd_c, crd_r, dim=-1)
        logits_cos = F.cosine_similarity(
            logits_c[0], logits_r[0], dim=-1
        ).mean().item()
        row = {
            "pool_index": pool_index,
            "source": source,
            "prefix_exact": prefix_exact,
            "ledger_length": int(ledger.shape[1]),
            "stage_a_lexical_tokens": len(stage_a["lexical_ids"]),
            "stage_b_lexical_tokens": len(stage_b["lexical_ids"]),
            "logits_cosine": logits_cos,
            "residual_cosine_mean": statistics.mean(resid_cos),
            "residual_cosine_min": min(resid_cos),
            "crd_cosine_mean": crd_cos.mean().item(),
            "crd_cosine_min": crd_cos.min().item(),
        }
        rows.append(row)
        print(f"[cache] {i + 1}/{args.n} {row}", flush=True)

    atomic_json(
        OUT / f"cache_replay_sentiment_seed{args.seed}_n{args.n}.json",
        {
            "task": "TweetEval sentiment negative-to-positive",
            "model": MODEL_ID,
            "dtype": "bfloat16",
            "seed": args.seed,
            "n": args.n,
            "protocol": "canonical token-identical cache vs full replay",
            "rows": rows,
        },
    )


def repetition_flag(text: str) -> bool:
    words = re.findall(r"[A-Za-z']+", text.lower())
    if not words:
        return True
    counts = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
    return max(counts.values()) / len(words) >= 0.5 and len(words) >= 8


def evaluate(args):
    model, tokenizer = load_model()
    classifier = pipeline(
        "sentiment-analysis",
        model=CLF_ID,
        device=0,
        truncation=True,
        max_length=512,
        local_files_only=True,
    )
    labels = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}
    texts, _ = load_texts()
    eval_texts = texts[: args.num_samples]
    vector = None
    vector_hash = None
    if args.vector:
        vector_path = Path(args.vector)
        vector = torch.load(vector_path, map_location="cpu", weights_only=False).float()
        vector_hash = sha256(vector_path)
    layer = get_layers(model)[args.layer]
    handle = None
    if vector is not None:
        v = vector[args.layer].to(model.device)

        def hook(_module, _inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            changed = h + args.alpha * v.to(dtype=h.dtype, device=h.device)
            return (
                (changed,) + output[1:]
                if isinstance(output, tuple)
                else changed
            )

        handle = layer.register_forward_hook(hook)

    rows = []
    try:
        for index, source in enumerate(eval_texts):
            messages = [
                {
                    "role": "user",
                    "content": (
                        "Rewrite the following tweet in a positive tone. "
                        f"Keep the same meaning.\n\n{source}"
                    ),
                }
            ]
            prompt = render(tokenizer, messages)
            ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
            with torch.no_grad():
                out = model.generate(
                    ids,
                    max_new_tokens=args.eval_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            text = tokenizer.decode(
                out[0][ids.shape[1] :], skip_special_tokens=True
            )
            pred = classifier(text)[0]
            label = labels.get(pred["label"], pred["label"])
            rows.append(
                {
                    "index": index,
                    "source": source,
                    "output": text,
                    "label": label,
                    "positive": int(label == "positive"),
                    "repetition_flag": repetition_flag(text),
                }
            )
            if (index + 1) % 25 == 0:
                print(f"[eval] {index + 1}/{len(eval_texts)}", flush=True)
    finally:
        if handle is not None:
            handle.remove()

    result = {
        "task": "TweetEval explicit positive-tone rewrite",
        "model": MODEL_ID,
        "classifier": CLF_ID,
        "condition": args.name,
        "vector_path": args.vector,
        "vector_sha256": vector_hash,
        "layer": args.layer,
        "alpha": args.alpha if vector is not None else 0.0,
        "n": len(rows),
        "positive_rate": sum(r["positive"] for r in rows) / len(rows),
        "repetition_rate": sum(r["repetition_flag"] for r in rows) / len(rows),
        "rows": rows,
    }
    path = OUT / "eval" / f"{args.name}_n{len(rows)}.json"
    atomic_json(path, result)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p_extract = sub.add_parser("extract")
    p_extract.add_argument("--condition", choices=CONDITIONS, required=True)
    p_extract.add_argument("--seed", type=int, default=42)
    p_extract.add_argument("--n", type=int, default=5)
    p_extract.add_argument("--max-new-tokens", type=int, default=80)

    p_cache = sub.add_parser("cachecheck")
    p_cache.add_argument("--seed", type=int, default=42)
    p_cache.add_argument("--n", type=int, default=5)
    p_cache.add_argument("--max-new-tokens", type=int, default=80)

    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--vector", default=None)
    p_eval.add_argument("--name", required=True)
    p_eval.add_argument("--layer", type=int, default=15)
    p_eval.add_argument("--alpha", type=float, default=1.5)
    p_eval.add_argument("--num-samples", type=int, default=500)
    p_eval.add_argument("--eval-new-tokens", type=int, default=80)

    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.command == "extract":
        extract_condition(args)
    elif args.command == "cachecheck":
        cache_check(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
