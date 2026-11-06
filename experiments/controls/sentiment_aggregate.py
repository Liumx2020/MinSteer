#!/usr/bin/env python3
"""Aggregate clean sentiment mechanism results with paired prompt bootstrap."""

from __future__ import annotations

import json
import random
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUT = HERE / "out"


def percentile(values, q):
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    weight = pos - lo
    return values[lo] * (1 - weight) + values[hi] * weight


def paired_delta(a, b, n_boot=10000, seed=42):
    by_a = {row["index"]: row["positive"] for row in a["rows"]}
    by_b = {row["index"]: row["positive"] for row in b["rows"]}
    ids = sorted(set(by_a) & set(by_b))
    diffs = [by_a[i] - by_b[i] for i in ids]
    rng = random.Random(seed)
    boots = [
        sum(diffs[rng.randrange(len(diffs))] for _ in diffs) / len(diffs)
        for _ in range(n_boot)
    ]
    return {
        "n": len(ids),
        "delta": sum(diffs) / len(diffs),
        "ci95": [percentile(boots, 0.025), percentile(boots, 0.975)],
        "self_only_success": sum(
            by_a[i] == 1 and by_b[i] == 0 for i in ids
        ),
        "other_only_success": sum(
            by_a[i] == 0 and by_b[i] == 1 for i in ids
        ),
    }


def main():
    paths = sorted((OUT / "eval").glob("plain_*_n500.json"))
    results = {
        json.loads(path.read_text(encoding="utf-8"))["condition"]:
        json.loads(path.read_text(encoding="utf-8"))
        for path in paths
    }
    rates = {
        name: {
            "positive_rate": value["positive_rate"],
            "repetition_rate": value["repetition_rate"],
            "vector_sha256": value["vector_sha256"],
        }
        for name, value in results.items()
    }

    comparisons = {}
    self42 = "plain_self_transcript_seed42_n5"
    if self42 in results:
        for other in (
            "plain_quoted_self_seed42_n5",
            "plain_quoted_mismatch_seed42_n5",
            "plain_twoseq_seed42_n5",
            "plain_baseline_alpha0",
        ):
            if other in results:
                comparisons[f"{self42}_minus_{other}"] = paired_delta(
                    results[self42], results[other]
                )

    multiseed = {}
    for family in ("self_transcript", "twoseq"):
        names = [
            f"plain_{family}_seed{seed}_n5" for seed in (42, 43, 44)
        ]
        values = [
            results[name]["positive_rate"] for name in names if name in results
        ]
        if values:
            multiseed[family] = {
                "values": values,
                "mean": statistics.mean(values),
                "sample_std": statistics.stdev(values)
                if len(values) > 1 else None,
            }

    cache_path = OUT / "cache_replay_sentiment_seed42_n5.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    cache_summary = {
        "n": len(cache["rows"]),
        "all_prefix_exact": all(x["prefix_exact"] for x in cache["rows"]),
        "min_crd_cosine": min(
            x["crd_cosine_min"] for x in cache["rows"]
        ),
        "min_residual_cosine": min(
            x["residual_cosine_min"] for x in cache["rows"]
        ),
        "min_logits_cosine": min(
            x["logits_cosine"] for x in cache["rows"]
        ),
    }

    summary = {
        "rates": rates,
        "paired_comparisons": comparisons,
        "multiseed": multiseed,
        "cache_replay": cache_summary,
    }
    (HERE / "SUMMARY.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
