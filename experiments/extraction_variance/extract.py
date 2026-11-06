#!/usr/bin/env python3
'Repeated sentiment extraction with matched evaluation inputs.'

from __future__ import annotations

import argparse
import json
import os
import random
from contextlib import contextmanager

import torch
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer
from summarize import METHOD_N


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SENT_DIR = os.path.join(ROOT, "sentiment")
TEXT_FILE = os.path.join(SENT_DIR, "data", "test_text.txt")
LABEL_FILE = os.path.join(SENT_DIR, "data", "test_labels.txt")
OUT_DIR = os.path.join(ROOT, "runs", "sentiment_multiseed", "vectors")
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
CLF_ID = "cardiffnlp/twitter-roberta-base-sentiment"
EXTRACTION_METHODS = [name for name in METHOD_N if not name.startswith('caa_class_')]


def load_data():
    with open(TEXT_FILE, encoding="utf-8") as f:
        texts = [x.rstrip("\n").strip('"') for x in f if x.strip()]
    with open(LABEL_FILE, encoding="utf-8") as f:
        labels = [int(x.strip()) for x in f if x.strip()]
    if len(texts) != len(labels):
        raise ValueError(f"text/label mismatch: {len(texts)} vs {len(labels)}")
    return list(zip(texts[:500], labels[:500])), texts[500:]


def apply_chat(tokenizer, messages):
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def sentiment_messages(text: str, tone: str):
    return [
        {
            "role": "system",
            "content": (
                "You rewrite social-media messages while preserving their core meaning."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Rewrite the following tweet in a {tone} tone while preserving its "
                f"meaning:\n\n{text}"
            ),
        },
    ]


def conversion_message(target_tone: str):
    return {
        "role": "user",
        "content": (
            f"Now rewrite the same message in a {target_tone} tone. "
            "Keep the same core meaning."
        ),
    }


def get_layer(model, layer: int):
    return model.model.layers[layer]


@contextmanager
def capture_layer(model, layer: int):
    captured = []

    def hook(_module, _inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        captured.append(h.detach().float().cpu())

    handle = get_layer(model, layer).register_forward_hook(hook)
    try:
        yield captured
    finally:
        handle.remove()


def stop_ids(tokenizer):
    ids = {tokenizer.eos_token_id}
    for token in ("<|eot_id|>", "<|im_end|>"):
        tid = tokenizer.convert_tokens_to_ids(token)
        if tid is not None and tid != tokenizer.unk_token_id:
            ids.add(tid)
    return ids


def generate_and_capture(
    model,
    tokenizer,
    messages,
    layer: int,
    max_new_tokens: int,
):
    """Greedy generation with resid_post recorded for the generated token positions."""
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
        with capture_layer(model, layer) as cap:
            with torch.no_grad():
                out = model(
                    next_id,
                    use_cache=True,
                    past_key_values=past,
                    return_dict=True,
                )
        if len(cap) != 1:
            raise RuntimeError(f"expected one capture, got {len(cap)}")
        generated_ids.append(token_id)
        generated_resids.append(cap[0][0, -1].clone())
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
    lex_resids = torch.stack([x[1] for x in lexical])
    text = tokenizer.decode(lex_ids, skip_special_tokens=True).strip()
    return {
        "text": text,
        "ids": lex_ids,
        "mean_resid": lex_resids.mean(dim=0),
        "last_resid": lex_resids[-1],
    }


def last_prompt_activations(model, tokenizer, texts, tone, layer, batch_size):
    acts = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        prompts = [
            apply_chat(tokenizer, sentiment_messages(text, tone)) for text in batch
        ]
        toks = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        ).to(model.device)
        with torch.no_grad():
            out = model(**toks, output_hidden_states=True, return_dict=True)
        # model layer L resid_post is hidden_states[L+1].
        h = out.hidden_states[layer + 1]
        assert tokenizer.padding_side == "left"
        acts.append(h[:, -1, :].detach().float().cpu())
    return torch.cat(acts, dim=0)


def normalize_unit(vector):
    return vector / vector.float().norm().clamp_min(1e-12)


def extract_one(method, seed, model, tokenizer, pool, args):
    n = METHOD_N[method]
    rng = random.Random(seed)
    indices = rng.sample(range(len(pool)), n)
    texts = [pool[i] for i in indices]
    records = []

    if method.startswith("minsteer"):
        deltas = []
        for i, text in enumerate(texts):
            print(f"[extract] {method} seed={seed} {i + 1}/{n}", flush=True)
            neg_messages = sentiment_messages(text, "negative")
            stage_a = generate_and_capture(
                model, tokenizer, neg_messages, args.layer, args.extract_tokens
            )
            stage_b_messages = neg_messages + [
                {"role": "assistant", "content": stage_a["text"]},
                conversion_message("positive"),
            ]
            stage_b = generate_and_capture(
                model, tokenizer, stage_b_messages, args.layer, args.extract_tokens
            )
            deltas.append(stage_b["mean_resid"] - stage_a["mean_resid"])
            records.append(
                {
                    "pool_index": indices[i],
                    "source": text,
                    "stage_a": stage_a["text"],
                    "stage_b": stage_b["text"],
                }
            )
        vector = torch.stack(deltas).mean(dim=0)

    else:
        neg = last_prompt_activations(
            model, tokenizer, texts, "negative", args.layer, args.batch_size
        )
        pos = last_prompt_activations(
            model, tokenizer, texts, "positive", args.layer, args.batch_size
        )
        if method.startswith("repe_"):
            relative = (neg - pos).numpy()
            pca = PCA(n_components=1, whiten=False).fit(relative)
            vector = torch.tensor(pca.components_[0], dtype=torch.float32)

            if ((pos @ vector) - (neg @ vector)).mean().item() < 0:
                vector = -vector
        else:
            raise ValueError(method)
        records = [
            {"pool_index": idx, "source": text}
            for idx, text in zip(indices, texts)
        ]

    raw_norm = float(vector.float().norm())
    vector = normalize_unit(vector.float())
    return vector, {
        "method": method,
        "n": n,
        "seed": seed,
        "pool_indices": indices,
        "raw_norm": raw_norm,
        "normalized_norm": float(vector.norm()),
        "storage": "unit",
        "records": records,
    }


@contextmanager
def steering_hook(model, layer, vector, alpha):
    vec = vector.to(model.device)

    def hook(_module, _inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        delta = alpha * vec.to(dtype=h.dtype, device=h.device)
        changed = h + delta
        return (changed,) + output[1:] if isinstance(output, tuple) else changed

    handle = get_layer(model, layer).register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def eval_one(method, seed, vector, model, tokenizer, classifier, eval_set, args):
    sign = 1.0
    outputs = []
    texts = [x[0] for x in eval_set]
    for start in range(0, len(texts), args.batch_size):
        batch = texts[start : start + args.batch_size]
        prompts = [
            apply_chat(tokenizer, sentiment_messages(text, "positive")) for text in batch
        ]
        toks = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        with steering_hook(model, args.layer, vector, sign * args.alpha):
            with torch.no_grad():
                out = model.generate(
                    **toks,
                    max_new_tokens=args.eval_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
        new = out[:, toks.input_ids.shape[1] :]
        outputs.extend(tokenizer.batch_decode(new, skip_special_tokens=True))
        print(
            f"[eval] {method} seed={seed} {min(start + len(batch), len(texts))}/{len(texts)}",
            flush=True,
        )

    preds = classifier(outputs, batch_size=32, truncation=True, max_length=512)
    label_map = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}
    pred_labels = [label_map.get(x["label"], x["label"].lower()) for x in preds]
    toward_pos = sum(x == "positive" for x in pred_labels) / len(pred_labels)
    return {
        "method": method,
        "n": METHOD_N[method],
        "seed": seed,
        "alpha": args.alpha,
        "toward_pos": toward_pos,
        "eval_size": len(eval_set),
        "prompt_protocol": "common_explicit_positive_tone",
        "eval_ids": list(range(len(eval_set))),
        "outputs": [
            {
                "id": i,
                "source": eval_set[i][0],
                "source_label": eval_set[i][1],
                "output": outputs[i],
                "pred_label": pred_labels[i],
            }
            for i in range(len(outputs))
        ],
    }


def vector_paths(method, seed):
    stem = f"{method}_seed{seed}"
    return (
        os.path.join(OUT_DIR, stem + ".pt"),
        os.path.join(OUT_DIR, stem + "_extract.json"),
        os.path.join(OUT_DIR, stem + "_eval.json"),
    )


def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16
    ).to("cuda:0")
    model.eval()
    return model, tokenizer

def main():
    global OUT_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["extract"], default="extract")
    parser.add_argument("--methods", nargs="+", choices=EXTRACTION_METHODS, default=EXTRACTION_METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument("--extract_tokens", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output-dir", default=OUT_DIR)
    args = parser.parse_args()
    OUT_DIR = os.path.abspath(args.output_dir)
    os.makedirs(OUT_DIR, exist_ok=True)
    _, pool = load_data()
    model, tokenizer = load_model_and_tokenizer()
    for method in args.methods:
        for seed in args.seeds:
            vec_path, meta_path, _ = vector_paths(method, seed)
            if os.path.exists(vec_path) or os.path.exists(meta_path):
                raise FileExistsError(vec_path)
            torch.manual_seed(seed)
            vector, meta = extract_one(method, seed, model, tokenizer, pool, args)
            torch.save(vector, vec_path)
            with open(meta_path, "w", encoding="utf-8") as stream:
                json.dump(meta, stream, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
