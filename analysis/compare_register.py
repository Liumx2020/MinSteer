#!/usr/bin/env python3
"""Summarize paired condition deltas with prompt-level bootstrap intervals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    report: dict[str, object] = {
        "bootstrap": {
            "unit": "source prompt",
            "resamples": args.resamples,
            "seed": args.seed,
            "interval": "percentile 95%",
        },
        "methods": {},
    }
    for path in args.inputs:
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data["rows"]
        by_condition: dict[str, dict[int, dict]] = {}
        for row in rows:
            by_condition.setdefault(str(row["condition"]), {})[
                int(row["source_prompt_idx"])
            ] = row
        baseline_key = next(
            key for key in by_condition if abs(float(key)) < 1e-12
        )
        baseline = by_condition[baseline_key]
        method: dict[str, object] = {"summaries": data["summary"], "deltas": {}}
        for condition, current in sorted(
            by_condition.items(), key=lambda item: float(item[0])
        ):
            if condition == baseline_key:
                continue
            ids = sorted(set(baseline) & set(current))
            entry: dict[str, object] = {"n_prompts": len(ids)}
            for metric in ("p_formal", "similarity_to_source"):
                delta = np.array(
                    [current[idx][metric] - baseline[idx][metric] for idx in ids]
                )
                samples = rng.choice(delta, size=(args.resamples, len(delta))).mean(1)
                entry[metric] = {
                    "mean_delta": float(delta.mean()),
                    "ci95": [float(x) for x in np.quantile(samples, [0.025, 0.975])],
                }
            method["deltas"][condition] = entry
        report["methods"][path.stem] = method

    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
