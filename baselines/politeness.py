#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_SCENARIOS = REPO / "prompts/politeness_scenarios.json"
DEFAULT_EVAL = REPO / "data/stanford99.json"
DEFAULT_OUTPUT = HERE / "outputs"
DEFAULT_VECTOR_DIR = HERE / "vectors"

MODEL_NAME = "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3"
SYSTEM_PROMPT = "You are rewriting requests. Keep the meaning, optionally adjust the tone."
EVAL_USER_TEMPLATE = (
    "Rewrite the following request, keeping the exact same meaning. "
    "Output only the rewritten text, 1-3 sentences:\n\n{text}"
)
JAPANESE_SYSTEM = "あなたは日本人のメール作成アシスタントです。指示に従い、メールを一通だけ書いてください。"


@dataclass(frozen=True)
class ExtractionRecord:
    scenario_id: int
    scenario: str
    formal_prompt: str
    casual_prompt: str
    formal_output: str
    casual_output: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(model_name: str, local_files_only: bool) -> tuple[Any, Any, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("This 8B experiment requires a CUDA GPU.")
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, local_files_only=local_files_only
    ).to(device)
    model.eval()
    return model, tokenizer, device


def apply_chat(tokenizer: Any, messages: list[dict[str, str]], *, generation: bool) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=generation)


def japanese_role_instruction(scenario: str, register: str) -> str:
    """Use matched scenario content; only the recipient/register instruction differs."""
    if register == "formal":
        role = (
            "あなたは従業員として上司にメールを書いています。相手への敬意を示し、"
            "謙譲語・尊敬語を自然に用いて、丁寧で改まった文体にしてください。"
        )
    elif register == "casual":
        role = (
            "あなたは従業員として親しい友人にメールを書いています。内容は同じまま、"
            "過度な敬語を避けた自然でくだけた文体にしてください。"
        )
    else:
        raise ValueError(register)
    return f"{role}\n\n【出力形式】件名、宛先、本文を含むメールを作成してください。\n\n【具体的な指示】\n{scenario}"


def greedy_generate(model: Any, tokenizer: Any, messages: list[dict[str, str]], max_new_tokens: int) -> str:
    device = next(model.parameters()).device
    prompt = apply_chat(tokenizer, messages, generation=True)
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    with torch.inference_mode():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(out[0, enc.input_ids.shape[1] :], skip_special_tokens=True).strip()


def last_token_resid_pre(model: Any, input_ids: torch.Tensor, layer: int) -> torch.Tensor:
    """Return the residual input to the selected transformer block at final token."""
    layers = model.model.layers
    if layer < 0 or layer >= len(layers):
        raise ValueError(f"layer={layer} is outside [0, {len(layers) - 1}]")
    captured: list[torch.Tensor | None] = [None]

    def hook(_module: Any, inputs: tuple[torch.Tensor, ...]) -> None:
        captured[0] = inputs[0][:, -1, :].detach().float().cpu().squeeze(0)

    handle = layers[layer].input_layernorm.register_forward_pre_hook(hook)
    try:
        with torch.inference_mode():
            model(input_ids, use_cache=False)
    finally:
        handle.remove()
    if captured[0] is None:
        raise RuntimeError("residual hook did not fire")
    return captured[0]


def full_sequence_last_resid(
    model: Any, tokenizer: Any, messages: list[dict[str, str]], answer: str, layer: int
) -> torch.Tensor:
    full_messages = [*messages, {"role": "assistant", "content": answer}]
    text = apply_chat(tokenizer, full_messages, generation=False)
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(next(model.parameters()).device)
    return last_token_resid_pre(model, ids, layer)


def l2_normalise(vector: torch.Tensor) -> torch.Tensor:
    return vector / vector.norm().clamp_min(1e-8)


def load_scenarios(path: Path, n: int, seed: int) -> list[dict[str, str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    items = [x for x in raw["prompts_for_detailed_analysis"] if x.get("scenario")]
    if n > len(items):
        raise ValueError(
            f"Requested n={n}, but the original audited keigo extraction file has only {len(items)} scenarios. "
            "Do not duplicate scenarios to claim a larger extraction budget."
        )
    rng = random.Random(seed)

    if n < len(items):
        items = rng.sample(items, n)
    return items


def extract(args: argparse.Namespace) -> None:
    if args.method not in {"caa", "repe"}:
        raise ValueError("extract --method must be caa or repe")
    set_seed(args.seed)
    scenarios = load_scenarios(args.scenarios, args.n, args.seed)
    model, tokenizer, _ = load_model(args.model, args.local_files_only)

    formal_states: list[torch.Tensor] = []
    casual_states: list[torch.Tensor] = []
    records: list[ExtractionRecord] = []
    for idx, item in enumerate(scenarios):
        scenario = item["scenario"]
        formal_user = japanese_role_instruction(scenario, "formal")
        casual_user = japanese_role_instruction(scenario, "casual")
        formal_messages = [{"role": "system", "content": JAPANESE_SYSTEM}, {"role": "user", "content": formal_user}]
        casual_messages = [{"role": "system", "content": JAPANESE_SYSTEM}, {"role": "user", "content": casual_user}]
        formal_out = greedy_generate(model, tokenizer, formal_messages, args.extract_max_new_tokens)
        casual_out = greedy_generate(model, tokenizer, casual_messages, args.extract_max_new_tokens)
        formal_states.append(full_sequence_last_resid(model, tokenizer, formal_messages, formal_out, args.layer))
        casual_states.append(full_sequence_last_resid(model, tokenizer, casual_messages, casual_out, args.layer))
        records.append(ExtractionRecord(idx, scenario, formal_user, casual_user, formal_out, casual_out))
        print(f"[{idx + 1}/{len(scenarios)}] collected independent formal/casual states", flush=True)

    formal = torch.stack(formal_states)  # [n, hidden]
    casual = torch.stack(casual_states)
    mean_difference = casual.mean(0) - formal.mean(0)
    if args.method == "caa":
        # CAA follows the existing toxicity implementation: normalise each paired
        # contrast before averaging, then normalise the resulting fixed vector.
        vector = l2_normalise(torch.stack([l2_normalise(d) for d in casual - formal]).mean(0))
        method_detail = "mean of per-pair L2-normalised (casual - formal) last-token residual contrasts"
    else:
        X = torch.cat([formal, casual], dim=0)
        X = X - X.mean(0, keepdim=True)
        _u, _s, vh = torch.linalg.svd(X, full_matrices=False)
        vector = vh[0]
        if torch.dot(vector, mean_difference) < 0:
            vector = -vector
        vector = l2_normalise(vector)
        method_detail = "PC1 over 2n independent formal/casual last-token states; sign aligned to casual - formal"

    args.vector_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.method}_n{args.n}_seed{args.seed}"
    vec_path = args.vector_dir / f"{stem}.pt"
    manifest_path = args.vector_dir / f"{stem}.json"
    torch.save(vector.to(torch.bfloat16), vec_path)
    manifest = {
        "created_at_utc": utc_now(),
        "method": args.method.upper(),
        "method_detail": method_detail,
        "vector_path": str(vec_path.resolve()),
        "vector_orientation": "positive alpha -> casual; negative alpha -> formal",
        "model": args.model,
        "layer": args.layer,
        "residual_type": "resid_pre",
        "readout": "last token after independently generated full email",
        "n_pairs": args.n,
        "seed": args.seed,
        "source_scenarios": str(args.scenarios.resolve()),
        "independence_note": "Each arm used a separate prompt and fresh generation; no KV cache or transcript is shared.",
        "records": [asdict(r) for r in records],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved vector: {vec_path}")
    print(f"saved manifest: {manifest_path}")


def hook_generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    vector: torch.Tensor,
    layer: int,
    strength: float,
    max_new_tokens: int,
) -> str:
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    delta = (strength * vector).to(device=device, dtype=next(model.parameters()).dtype)

    def pre_hook(_module: Any, inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        # Clone prevents an in-place mutation from leaking into cached activations.
        hidden = inputs[0].clone()
        hidden[:, -1, :] += delta
        return (hidden, *inputs[1:])

    handle = model.model.layers[layer].input_layernorm.register_forward_pre_hook(pre_hook)
    try:
        with torch.inference_mode():
            output = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
    finally:
        handle.remove()
    return tokenizer.decode(output[0, enc.input_ids.shape[1] :], skip_special_tokens=True).strip()


def maybe_load_formality_classifier(device: torch.device, enabled: bool) -> Any | None:
    if not enabled:
        return None
    try:
        from transformers import pipeline

        return pipeline(
            "text-classification",
            model="s-nlp/roberta-base-formality-ranker",
            device=0 if device.type == "cuda" else -1,
            truncation=True,
            max_length=128,
        )
    except Exception as exc:
        print(f"WARNING: Formality classifier unavailable; outputs will be saved unscored: {exc}", file=sys.stderr)
        return None


def formality_probability(text: str, classifier: Any | None) -> float | None:
    if classifier is None or not text.strip():
        return None
    result = classifier(text[:512])[0]
    label = str(result["label"]).lower()
    score = float(result["score"])
    return score if label == "formal" else 1.0 - score


def chunks(items: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def evaluate(args: argparse.Namespace) -> None:
    if not args.vector.exists():
        raise FileNotFoundError(args.vector)
    set_seed(args.seed)
    raw = json.loads(args.eval_json.read_text(encoding="utf-8"))
    samples = raw["samples"][: args.max_prompts] if args.max_prompts else raw["samples"]
    if not samples:
        raise ValueError("No evaluation samples found")
    vector = torch.load(args.vector, map_location="cpu", weights_only=True).float()
    if vector.ndim != 1:
        raise ValueError(f"Expected a one-layer vector [hidden], got {tuple(vector.shape)}")
    model, tokenizer, device = load_model(args.model, args.local_files_only)
    clf = maybe_load_formality_classifier(device, not args.no_formality_score)

    all_rows: list[dict[str, Any]] = []
    for strength in args.strengths:
        rows: list[dict[str, Any]] = []
        for position, sample in enumerate(samples, start=1):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": EVAL_USER_TEMPLATE.format(text=sample["orig_text"])},
            ]
            prompt = apply_chat(tokenizer, messages, generation=True)
            output = hook_generate(model, tokenizer, prompt, vector, args.layer, strength, args.max_new_tokens)
            rows.append(
                {
                    "source_prompt_idx": position - 1,
                    "name": sample.get("name", f"stanford_{position - 1}"),
                    "orig_text": sample["orig_text"],
                    "orig_binary": sample.get("orig_binary"),
                    "strength": strength,
                    "output": output,
                }
            )
            print(f"strength={strength:+.3f} [{position}/{len(samples)}]", flush=True)

        if clf is not None:
            for group in chunks(rows, 16):
                texts = [x["output"][:512] if x["output"].strip() else " " for x in group]
                scored = clf(texts)
                for row, pred in zip(group, scored):
                    label = str(pred["label"]).lower()
                    score = float(pred["score"])
                    row["p_formal"] = score if label == "formal" else 1.0 - score
        all_rows.extend(rows)

    grouped: dict[str, list[float]] = {}
    for row in all_rows:
        if row.get("p_formal") is not None:
            grouped.setdefault(str(row["strength"]), []).append(float(row["p_formal"]))
    means = {strength: sum(values) / len(values) for strength, values in grouped.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at_utc": utc_now(),
        "tag": args.tag,
        "method_vector": str(args.vector.resolve()),
        "model": args.model,
        "layer": args.layer,
        "residual_type": "resid_pre",
        "hook_site": "model.model.layers[layer].input_layernorm pre-hook",
        "decoding": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
        "evaluation_source": str(args.eval_json.resolve()),
        "n_prompts": len(samples),
        "strengths": args.strengths,
        "mean_p_formal": means,
        "classifier": "s-nlp/roberta-base-formality-ranker" if clf is not None else None,
        "rows": all_rows,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved evaluation: {args.output}")
    if means:
        print("mean P(formal):", json.dumps(means, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", default=MODEL_NAME)
    common.add_argument("--layer", type=int, default=15)
    common.add_argument("--seed", type=int, default=42)
    common.add_argument("--local-files-only", action="store_true", help="fail instead of downloading a missing model")

    e = sub.add_parser("extract", parents=[common])
    e.add_argument("--method", choices=["caa", "repe"], required=True)
    e.add_argument("--n", type=int, default=5)
    e.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS)
    e.add_argument("--vector-dir", type=Path, default=DEFAULT_VECTOR_DIR)
    e.add_argument("--extract-max-new-tokens", type=int, default=180)
    e.set_defaults(func=extract)

    v = sub.add_parser("evaluate", parents=[common])
    v.add_argument("--vector", type=Path, required=True)
    v.add_argument("--tag", required=True)
    v.add_argument("--eval-json", type=Path, default=DEFAULT_EVAL)
    v.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "politeness_baseline_eval.json")
    v.add_argument("--strengths", nargs="+", type=float, default=[-0.25, 0.0, 0.25])
    v.add_argument("--max-new-tokens", type=int, default=150)
    v.add_argument("--max-prompts", type=int, default=0, help="0 means the full audited 99-prompt set")
    v.add_argument("--no-formality-score", action="store_true")
    v.set_defaults(func=evaluate)
    return p


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
