#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import argparse
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)


import run_all_toxicity_eval as ev

ev._lazy_imports()
torch = ev.torch
AutoTokenizer = ev.AutoTokenizer
AutoModelForCausalLM = ev.AutoModelForCausalLM


def _messages_stageB_standalone(toxic_text: str):
    system = (
        "You are a communication style consultant. "
        "Your job is to help rephrase messages in different tones while preserving the core meaning."
    )
    user = (
        f'Rephrase the following in a professional, polite, and appropriate way:\n'
        f'"{toxic_text}"\n\n'
        "Output only the rewritten sentence:"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _apply_chat_template_full(tokenizer, messages, config_path, add_generation_prompt=True):
    return ev._apply_chat_template(tokenizer, messages, path=config_path)


def get_last_token_residuals(model, tokenizer, input_ids, device, residual_key="resid_post"):
    layers = ev.get_layers(model)
    num_layers = len(layers)
    captured = [None] * num_layers

    def hook_factory(layer_idx):
        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            if h is not None:
                captured[layer_idx] = h[:, -1, :].detach().to("cpu", copy=True).float()
        return hook

    hooks = []
    for i, layer in enumerate(layers):
        hooks.append(layer.register_forward_hook(hook_factory(i)))

    with torch.no_grad():
        model(input_ids.to(device), use_cache=False)

    for h in hooks:
        try:
            h.remove()
        except Exception:
            pass

    if not all(captured[i] is not None for i in range(num_layers)):
        return None
    return torch.stack(captured, dim=0)


def main():
    parser = argparse.ArgumentParser(description='Extract twoseq last.')
    parser.add_argument("--alias", type=str, default="Llama-3.1-8B",
                        help='Alias.')
    parser.add_argument("--n_pairs", type=int, default=2,
                        help='Number of extraction pairs.')
    parser.add_argument("--pairing", type=str, choices=("same", "cross"), default="cross",
                        help='Pairing.')
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = None
    for c in ev.MODELS_CONFIG:
        if c["alias"] == args.alias:
            cfg = c
            break
    if not cfg:
        print(f"未找到模型 alias: {args.alias}")
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"加载模型: {cfg['path']}")
    tokenizer = AutoTokenizer.from_pretrained(cfg["path"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"], torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    eot_id = tokenizer.eos_token_id

    cache_path = os.path.join(
        ev.OUTPUT_DIR,
        f"paradetox_splits_seed{ev.PD_SEED}_train{ev.N_PD_TRAIN_CANDS}_eval{ev.N_PD_EVAL}.json",
    )
    if not os.path.exists(cache_path):
        cache_path = None
    train_pairs, _ = ev.load_paradetox_splits(
        n_train_cands=min(ev.N_PD_TRAIN_CANDS, 100),
        n_eval=10,
        seed=args.seed,
        cache_path=cache_path,
    )

    all_diffs = []
    for idx in range(min(args.n_pairs, len(train_pairs))):
        toxic1, _ = train_pairs[idx]
        if args.pairing == "same":
            toxic2 = toxic1
        else:
            toxic2_idx = (idx + 1) % len(train_pairs)
            toxic2, _ = train_pairs[toxic2_idx]


        msgs_a = ev._residual_chat_messages_stageA(toxic1)
        prompt_a = _apply_chat_template_full(tokenizer, msgs_a, cfg.get("path"))
        ids_a = tokenizer(prompt_a, return_tensors="pt").to(device).input_ids
        gen_a = ev._capture_residuals_tokenwise(
            model, tokenizer, ids_a,
            past_key_values=None,
            max_new_tokens=120,
            sampling_method="argmax",
            force_append_eot_id=eot_id,
        )
        if ev._is_refusal(gen_a["generated_text"]):
            print(f"  段1 拒绝，跳过 pair {idx}")
            continue
        full_msgs_1 = msgs_a + [{"role": "assistant", "content": gen_a["generated_text"]}]
        try:
            full_text_1 = tokenizer.apply_chat_template(
                full_msgs_1, tokenize=False, add_generation_prompt=False
            )
        except Exception:
            full_text_1 = prompt_a + gen_a["generated_text"]
        seq_1 = tokenizer(full_text_1, return_tensors="pt", add_special_tokens=False).to(device).input_ids
        r1 = get_last_token_residuals(model, tokenizer, seq_1, device)
        if r1 is None:
            continue


        msgs_b = _messages_stageB_standalone(toxic2)
        prompt_b = _apply_chat_template_full(tokenizer, msgs_b, cfg.get("path"))
        ids_b = tokenizer(prompt_b, return_tensors="pt").to(device).input_ids
        gen_b = ev._capture_residuals_tokenwise(
            model, tokenizer, ids_b,
            past_key_values=None,
            max_new_tokens=120,
            sampling_method="argmax",
            force_append_eot_id=eot_id,
        )
        if ev._is_refusal(gen_b["generated_text"]):
            print(f"  段2 拒绝，跳过 pair {idx}")
            continue
        full_msgs_2 = msgs_b + [{"role": "assistant", "content": gen_b["generated_text"]}]
        try:
            full_text_2 = tokenizer.apply_chat_template(
                full_msgs_2, tokenize=False, add_generation_prompt=False
            )
        except Exception:
            full_text_2 = prompt_b + gen_b["generated_text"]
        seq_2 = tokenizer(full_text_2, return_tensors="pt", add_special_tokens=False).to(device).input_ids
        r2 = get_last_token_residuals(model, tokenizer, seq_2, device)
        if r2 is None:
            continue

        d = r2 - r1
        all_diffs.append(d)
        print(f"  pair {idx}: diff shape {d.shape}")

    if not all_diffs:
        print("未得到有效差分，退出")
        return
    vec = torch.stack(all_diffs).mean(dim=0)
    if vec.dim() == 3:
        vec = vec.squeeze(1)
    vec = ev._normalize_vector_per_layer(vec)

    out_dir = os.path.join(ev.VECTOR_DIR, args.alias.replace(".", "_"))
    os.makedirs(out_dir, exist_ok=True)
    suffix = "last_token_same" if args.pairing == "same" else "last_token"
    out_path = os.path.join(out_dir, f"vector_paradetox_{suffix}.pt")
    torch.save(vec, out_path)
    print(f"已保存 ({args.pairing}): {out_path}  shape={vec.shape}")


if __name__ == "__main__":
    main()
