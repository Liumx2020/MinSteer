#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


OFFICIAL_COMMIT = "e84a4cdce122a9934bee1260f3fd3ddf8adc19a7"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_layers(text: str) -> list[int]:
    out: list[int] = []
    for segment in text.split(","):
        if "-" in segment:
            lo, hi = map(int, segment.split("-", 1))
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(segment))
    return sorted(set(out))


class OfficialMLPOutputSteering:
    """Exact hook-site and last-token behavior used by official vector.py."""

    def __init__(self, vector: torch.Tensor):
        self.vector = vector

    def __call__(self, _module: torch.nn.Module, _inputs: tuple[object, ...], output: torch.Tensor) -> torch.Tensor:
        # Official code mutates outp[:, -1, :] for a [batch, seq, hidden] MLP output.
        if output.ndim == 3:
            output[:, -1, :] += self.vector.to(device=output.device, dtype=output.dtype)
        elif output.ndim == 2:
            output[-1, :] += self.vector.to(device=output.device, dtype=output.dtype)
        else:
            raise RuntimeError(f"Unexpected MLP output shape: {tuple(output.shape)}")
        return output


def records(path: Path) -> Iterable[dict]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Local Qwen2.5-7B-Instruct model path")
    p.add_argument("--vector", type=Path, required=True)
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--output-jsonl", type=Path, required=True)
    p.add_argument("--layers", default="18-23", help="Official README's formality default")
    p.add_argument("--alpha", type=float, required=True)
    p.add_argument("--max-new-tokens", type=int, default=150)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None, help="Smoke test only; never report as the 99-prompt result")
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This minimal runner requires CUDA.")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    layers = parse_layers(args.layers)
    vector_data = torch.load(args.vector, map_location="cpu", weights_only=False)
    expected = [f"model.layers.{i}.mlp" for i in layers]
    missing = [name for name in expected if name not in vector_data]
    if missing:
        raise KeyError(f"Official vector lacks requested layers: {missing}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, torch_dtype=dtype, device_map="auto"
    )
    model.eval()
    modules = dict(model.named_modules())
    if any(name not in modules for name in expected):
        raise KeyError("Qwen model module names do not match official vector keys")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "kind": "metadata",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_repo_commit": OFFICIAL_COMMIT,
        "official_semantics": "normalized vector * alpha, MLP-output forward hook, last token only",
        "model": args.model,
        "vector": str(args.vector),
        "vector_sha256": sha256(args.vector),
        "prompts": str(args.prompts),
        "prompts_sha256": sha256(args.prompts),
        "layers": layers,
        "alpha": args.alpha,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
    }
    with args.output_jsonl.open("w", encoding="utf-8") as out:
        out.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        for count, item in enumerate(records(args.prompts)):
            if args.limit is not None and count >= args.limit:
                break
            text = tokenizer.apply_chat_template(item["messages"], tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            handles = []
            if args.alpha != 0.0:
                for name in expected:
                    unit = vector_data[name].float()
                    norm = torch.linalg.norm(unit)
                    if not torch.isfinite(norm) or norm.item() == 0:
                        raise ValueError(f"Invalid official vector at {name}")
                    handles.append(modules[name].register_forward_hook(
                        OfficialMLPOutputSteering((unit / norm) * args.alpha)
                    ))
            try:
                with torch.inference_mode():
                    output = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                        repetition_penalty=1.1,
                    )
            finally:
                for handle in handles:
                    handle.remove()
            generated = tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            out.write(json.dumps({
                "kind": "result", "index": item["index"], "name": item["name"],
                "source_prompt_idx": item["source_prompt_idx"], "orig_text": item["orig_text"],
                "generated_text": generated, "alpha": args.alpha,
            }, ensure_ascii=False) + "\n")
            out.flush()
            print(f"[{count + 1}] {item['name']}", flush=True)


if __name__ == "__main__":
    main()
