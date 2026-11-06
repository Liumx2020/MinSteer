#!/usr/bin/env python3

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import sentiment as core  # noqa: E402
import toxicity as tc  # noqa: E402

tc.OUT = HERE / "out_toxicity_prompt"


def canonical_polite_messages(text):
    return [
        {
            "role": "system",
            "content": (
                "You are a communication style consultant. Your job is to help "
                "rephrase messages in different tones while preserving the core meaning."
            ),
        },
        {
            "role": "user",
            "content": (
                f'A user wrote this message:\n"{text}"\n\n'
                "Now rephrase the same core message in a professional, polite, and "
                "appropriate way suitable for a public or workplace setting. The goal "
                "is to express the same idea but in a respectful tone:"
            ),
        },
    ]


def extract_twoseq_canonical(args):
    model, tokenizer = core.load_model()
    data = tc.load_split()["train_pairs_candidates"]
    shuffled = list(data)
    random.Random(args.seed).shuffle(shuffled)
    selected = shuffled[: args.n]
    diffs = []
    records = []
    for index, item in enumerate(selected):
        arm1 = core.clean_generate_fixed(
            model, tokenizer, tc.stage_a_messages(item["toxic"]), args.max_new_tokens
        )
        arm2 = core.clean_generate_fixed(
            model,
            tokenizer,
            canonical_polite_messages(item["toxic"]),
            args.max_new_tokens,
        )
        raw = arm2["mean_resid"] - arm1["mean_resid"]
        diffs.append(tc.normalize_layers(raw))
        records.append(
            {
                "toxic": item["toxic"],
                "neutral": item["neutral"],
                "arm1_casual_independent": arm1["text"],
                "arm2_polite_independent": arm2["text"],
                "arm1_refusal": tc.refusal(arm1["text"]),
                "arm2_refusal": tc.refusal(arm2["text"]),
                "raw_layer_norms": raw.norm(dim=-1).tolist(),
            }
        )
        print(f"[extract-twoseq-canonical] seed={args.seed} {index + 1}/{args.n}", flush=True)
    vector = tc.normalize_layers(torch.stack(diffs).mean(dim=0))
    run = tc.OUT / f"extract_twoseq_seed{args.seed}_n{args.n}"
    run.mkdir(parents=True, exist_ok=True)
    path = run / "vector.pt"
    torch.save(vector, path)
    core.atomic_json(
        run / "manifest.json",
        {
            "task": (
                "ParaDetox independent TwoSeq, arm2 = canonical Stage-B-parallel "
                "polite prompt (NOT the eval prompt)"
            ),
            "arm2_template": "canonical Stage-B-parallel polite instruction",
            "seed": args.seed,
            "n": args.n,
            "model": core.MODEL_ID,
            "readout": "generated lexical-token mean resid_post",
            "aggregation": "per-pair per-layer normalize, mean, final per-layer normalize",
            "vector_sha256": core.sha256(path),
            "records": records,
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--stage", choices=["extract", "eval", "both"], default="both")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument("--extract-max-new-tokens", type=int, default=80)
    parser.add_argument(
        "--doses",
        default="0.4",
        help="comma-separated reverse-arm doses to evaluate",
    )
    args = parser.parse_args()

    tc.DOSES = tuple(float(x) for x in args.doses.split(","))
    tc.OUT.mkdir(parents=True, exist_ok=True)

    if args.stage in ("extract", "both"):
        extract_twoseq_canonical(
            SimpleNamespace(
                seed=args.seed, n=args.n, max_new_tokens=args.extract_max_new_tokens
            )
        )
    if args.stage in ("eval", "both"):
        tc.evaluate(
            SimpleNamespace(
                seed=args.seed,
                n=args.n,
                variant="twoseq",
                batch_size=args.batch_size,
                window=args.window,
                max_new_tokens=args.max_new_tokens,
            )
        )


if __name__ == "__main__":
    main()
