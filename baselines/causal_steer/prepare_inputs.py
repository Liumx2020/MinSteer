#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SYSTEM = "You rewrite requests. Preserve the original meaning and make no factual additions."
USER = "Rewrite the request below in clear English. Output only the rewrite, in 1-3 sentences.\n\n{source}"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source-json", type=Path, required=True)
    p.add_argument("--output-jsonl", type=Path, required=True)
    p.add_argument("--manifest-json", type=Path, required=True)
    args = p.parse_args()

    source = json.loads(args.source_json.read_text(encoding="utf-8"))
    rows = source["samples"]
    if len(rows) != 99:
        raise ValueError(f"Expected the frozen 99 prompts, found {len(rows)}")
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for index, row in enumerate(rows):
            record = {
                "index": index,
                "name": row["name"],
                "source_prompt_idx": index,
                "orig_text": row["orig_text"],
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": USER.format(source=row["orig_text"])},
                ],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    manifest = {
        "n_prompts": len(rows),
        "source_json": str(args.source_json),
        "source_sha256": sha256(args.source_json),
        "prompt_jsonl": str(args.output_jsonl),
        "prompt_sha256": sha256(args.output_jsonl),
        "system_prompt": SYSTEM,
        "user_template": USER,
        "purpose": "Qwen2.5-7B-Instruct Causal-Steer formality comparison; same 99 English source texts as existing politeness evaluation",
    }
    args.manifest_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
