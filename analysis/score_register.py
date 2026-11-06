#!/usr/bin/env python3
"""Score Causal-Steer JSONL and Swallow baseline JSON outputs uniformly."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline


FORMALITY_MODEL = "s-nlp/roberta-base-formality-ranker"
SIM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def formality_probability(result: dict) -> float:
    label = str(result["label"]).lower()
    score = float(result["score"])
    return score if label == "formal" else 1.0 - score


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    b = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    return np.sum(a * b, axis=1)


def load_rows(path: Path) -> tuple[dict, list[dict]]:
    if path.suffix == ".jsonl":
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        metadata = next((x for x in records if x.get("kind") == "metadata"), {})
        rows = [
            {
                "source_prompt_idx": x["source_prompt_idx"],
                "source": x["orig_text"],
                "output": x["generated_text"],
                "condition": str(x["alpha"]),
            }
            for x in records
            if x.get("kind") == "result"
        ]
        return metadata, rows
    data = json.loads(path.read_text(encoding="utf-8"))
    if "samples" in data and "rows" not in data:
        selected = {"-1.0", "0.0", "1.0"}
        rows = [
            {
                "source_prompt_idx": index,
                "source": sample["orig_text"],
                "output": output,
                "condition": str(condition),
            }
            for index, sample in enumerate(data["samples"])
            for condition, output in sample["outputs"].items()
            if str(condition) in selected
        ]
        return data.get("metadata", {}), rows
    rows = [
        {
            "source_prompt_idx": x["source_prompt_idx"],
            "source": x["orig_text"],
            "output": x["output"],
            "condition": str(x["strength"]),
        }
        for x in data["rows"]
    ]
    return {k: v for k, v in data.items() if k != "rows"}, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    pipe_device = 0 if args.device.startswith("cuda") else -1
    formality = pipeline(
        "text-classification",
        model=FORMALITY_MODEL,
        device=pipe_device,
        truncation=True,
        max_length=128,
        model_kwargs={"local_files_only": True},
    )
    embedder = SentenceTransformer(
        SIM_MODEL, device=args.device, local_files_only=True
    )

    summaries = []
    all_scored = []
    for path in args.inputs:
        metadata, rows = load_rows(path)
        sources = [x["source"] for x in rows]
        outputs = [x["output"] or "" for x in rows]
        formal_raw = formality(outputs, batch_size=32)
        formal = np.array([formality_probability(x) for x in formal_raw])
        source_emb = embedder.encode(
            sources, batch_size=32, convert_to_numpy=True
        )
        output_emb = embedder.encode(
            outputs, batch_size=32, convert_to_numpy=True
        )
        similarity = cosine_rows(source_emb, output_emb)
        valid = np.array(
            [
                bool(re.search(r"[A-Za-z]", text))
                and len(set(re.findall(r"[A-Za-z']+", text.lower()))) > 1
                for text in outputs
            ],
            dtype=float,
        )
        summary = {
            "input": str(path),
            "n": len(rows),
            "condition": sorted(set(x["condition"] for x in rows)),
            "mean_p_formal": float(formal.mean()),
            "mean_similarity_to_source": float(similarity.mean()),
            "valid_rate": float(valid.mean()),
            "mean_words": float(np.mean([len(text.split()) for text in outputs])),
            "input_metadata": metadata,
        }
        summaries.append(summary)
        for row, p_formal, sim, is_valid in zip(
            rows, formal, similarity, valid
        ):
            all_scored.append(
                {
                    **row,
                    "input": str(path),
                    "p_formal": float(p_formal),
                    "similarity_to_source": float(sim),
                    "valid": bool(is_valid),
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {"models": {"formality": FORMALITY_MODEL, "similarity": SIM_MODEL},
             "summary": summaries, "rows": all_scored},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    for row in summaries:
        print(json.dumps({k: v for k, v in row.items() if k != "input_metadata"}))


if __name__ == "__main__":
    main()
