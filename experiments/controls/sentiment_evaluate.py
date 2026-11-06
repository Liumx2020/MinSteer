#!/usr/bin/env python3
"""Evaluate sentiment transcript-control directions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import pipeline

import sentiment as sc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vector", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument("--alpha", type=float, default=1.5)
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    model, tokenizer = sc.load_model()
    tokenizer.padding_side = "left"
    classifier = pipeline(
        "sentiment-analysis",
        model=sc.CLF_ID,
        device=0,
        truncation=True,
        max_length=512,
    )
    label_map = {
        "LABEL_0": "negative",
        "LABEL_1": "neutral",
        "LABEL_2": "positive",
    }
    vector_path = Path(args.vector)
    vector = torch.load(
        vector_path, map_location="cpu", weights_only=False
    ).float()
    layer_vector = vector[args.layer].to(model.device)

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        changed = hidden + args.alpha * layer_vector.to(
            dtype=hidden.dtype, device=hidden.device
        )
        return (
            (changed,) + output[1:]
            if isinstance(output, tuple)
            else changed
        )

    handle = sc.get_layers(model)[args.layer].register_forward_hook(hook)
    texts, _ = sc.load_texts()
    eval_texts = texts[: args.num_samples]
    rows = []
    try:
        for start in range(0, len(eval_texts), args.batch_size):
            sources = eval_texts[start : start + args.batch_size]
            prompts = [
                "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                f"Rewrite the following tweet: {source}"
                "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                for source in sources
            ]
            encoded = tokenizer(
                prompts, return_tensors="pt", padding=True
            ).to(model.device)
            with torch.no_grad():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            input_width = encoded.input_ids.shape[1]
            outputs = tokenizer.batch_decode(
                generated[:, input_width:], skip_special_tokens=True
            )
            predictions = classifier(
                outputs, batch_size=32, truncation=True, max_length=512
            )
            for offset, (source, output, prediction) in enumerate(
                zip(sources, outputs, predictions)
            ):
                label = label_map.get(
                    prediction["label"], prediction["label"]
                )
                rows.append(
                    {
                        "index": start + offset,
                        "source": source,
                        "output": output,
                        "label": label,
                        "positive": int(label == "positive"),
                        "repetition_flag": sc.repetition_flag(output),
                    }
                )
            print(
                f"[eval] {min(start + args.batch_size, len(eval_texts))}/"
                f"{len(eval_texts)}",
                flush=True,
            )
    finally:
        handle.remove()

    result = {
        "task": "TweetEval Table-4 plain rewrite",
        "model": sc.MODEL_ID,
        "classifier": sc.CLF_ID,
        "condition": args.name,
        "vector_path": str(vector_path),
        "vector_sha256": sc.sha256(vector_path),
        "layer": args.layer,
        "alpha": args.alpha,
        "n": len(rows),
        "batch_size": args.batch_size,
        "positive_rate": sum(r["positive"] for r in rows) / len(rows),
        "repetition_rate": sum(r["repetition_flag"] for r in rows) / len(rows),
        "rows": rows,
    }
    output_path = sc.OUT / "eval" / f"{args.name}_n{len(rows)}.json"
    sc.atomic_json(output_path, result)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
