#!/usr/bin/env python
# -*- coding: utf-8 -*-
from release_paths import project_path

import os
import sys
import json
import argparse
import torch
import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE_DIR = project_path('politeness')
SCRIPT_FOLDER = os.path.join(BASE_DIR, "extract_cached")
SHORT_EMAILS_INPUT = project_path('prompts/politeness_scenarios.json')
OUTPUT_DIR = SCRIPT_FOLDER


sys.path.insert(0, BASE_DIR)
import importlib.util
spec = importlib.util.spec_from_file_location(
    "residual_extract",
    os.path.join(BASE_DIR, "extract_cached.py")
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
create_initial_user_prompt = mod.create_initial_user_prompt
token_wise_generation_with_residuals = mod.token_wise_generation_with_residuals

SYSTEM_PROMPT = "あなたは日本人のメール作成アシスタントです。以下の指示に従って、適切なメールを一通だけ書いてください。"


def get_last_token_residuals(model, tokenizer, input_ids, device, residual_types=("resid_pre", "resid_mid", "resid_post")):
    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    captured = {k: [None] * num_layers for k in residual_types}

    def hook_factory(key, layer_idx, is_pre_or_mid):
        def hook(module, inputs, output):

            x = inputs[0] if is_pre_or_mid else (output[0] if isinstance(output, tuple) else output)
            captured[key][layer_idx] = x[:, -1, :].detach().to("cpu", copy=True).float()
        return hook

    hooks = []
    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.input_layernorm.register_forward_hook(hook_factory("resid_pre", i, True)))
        hooks.append(layer.post_attention_layernorm.register_forward_hook(hook_factory("resid_mid", i, True)))
        hooks.append(layer.register_forward_hook(hook_factory("resid_post", i, False)))

    with torch.no_grad():
        model(input_ids.to(device), use_cache=False)

    for h in hooks:
        h.remove()

    result = {}
    for key in residual_types:
        if all(captured[key][i] is not None for i in range(num_layers)):
            result[key] = torch.stack(captured[key], dim=0)  # [num_layers, H]
        else:
            result[key] = None
    return result


def main():
    parser = argparse.ArgumentParser(description='Extract twoseq last.')
    parser.add_argument("--model", type=str, default="swallow", choices=["swallow", "elyza"])
    parser.add_argument("--prompt_index", type=int, default=0, help='Prompt index.')
    parser.add_argument("--layer", type=int, default=15, help='Zero-based layer index.')
    args = parser.parse_args()

    try:
        with open(SHORT_EMAILS_INPUT, "r", encoding="utf-8") as f:
            data = json.load(f)
        prompts = [p for p in data["prompts_for_detailed_analysis"] if p.get("role_pair") == "従業員→上司"]
    except FileNotFoundError:
        print(f"❌ 找不到: {SHORT_EMAILS_INPUT}")
        return
    if not prompts:
        print("❌ 没有 従業員→上司 的 prompt")
        return

    prompt_data = prompts[args.prompt_index]
    scenario = prompt_data["scenario"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3" if args.model == "swallow" else "elyza/Llama-3-ELYZA-JP-8B"

    print(f"🚀 加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    eot_id = tokenizer.eos_token_id


    sender_1, receiver_1 = "従業員", "上司"
    user_prompt_1 = create_initial_user_prompt(scenario, sender_1, receiver_1)
    messages_1 = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt_1},
    ]
    prompt_text_1 = tokenizer.apply_chat_template(messages_1, tokenize=False, add_generation_prompt=True)
    prompt_ids_1 = tokenizer(prompt_text_1, return_tensors="pt").to(device)["input_ids"]

    gen_result_1 = token_wise_generation_with_residuals(
        model, tokenizer, input_ids=prompt_ids_1, max_new_tokens=200,
        sampling_method="argmax", force_append_eot_id=eot_id,
    )
    response1 = gen_result_1["generated_text"]
    print(f"✅ 阶段1 生成 (従業員→上司): {len(response1)} 字")


    receiver_2 = "友人"
    user_prompt_2 = create_initial_user_prompt(scenario, sender_1, receiver_2)
    messages_2 = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt_2},
    ]
    prompt_text_2 = tokenizer.apply_chat_template(messages_2, tokenize=False, add_generation_prompt=True)
    prompt_ids_2 = tokenizer(prompt_text_2, return_tensors="pt").to(device)["input_ids"]

    gen_result_2 = token_wise_generation_with_residuals(
        model, tokenizer, input_ids=prompt_ids_2, max_new_tokens=200,
        sampling_method="argmax", force_append_eot_id=eot_id,
    )
    response2 = gen_result_2["generated_text"]
    print(f"✅ 阶段2 生成 (従業員→友人): {len(response2)} 字")


    full_messages_1 = messages_1 + [{"role": "assistant", "content": response1}]
    full_messages_2 = messages_2 + [{"role": "assistant", "content": response2}]
    full_text_1 = tokenizer.apply_chat_template(
        full_messages_1, tokenize=False, add_generation_prompt=False
    )
    full_text_2 = tokenizer.apply_chat_template(
        full_messages_2, tokenize=False, add_generation_prompt=False
    )
    seq_ids_1 = tokenizer(full_text_1, return_tensors="pt", add_special_tokens=False).to(device)["input_ids"]
    seq_ids_2 = tokenizer(full_text_2, return_tensors="pt", add_special_tokens=False).to(device)["input_ids"]

    print("📐 抽取序列1 末尾残差 (Prompt1+response1)...")
    res_1 = get_last_token_residuals(model, tokenizer, seq_ids_1, device)
    print("📐 抽取序列2 末尾残差 (Prompt2+response2)...")
    res_2 = get_last_token_residuals(model, tokenizer, seq_ids_2, device)


    diff_vectors = {}
    for key in ("resid_pre", "resid_mid", "resid_post"):
        if res_1.get(key) is not None and res_2.get(key) is not None:
            diff_vectors[key] = (res_2[key] - res_1[key]).to(torch.bfloat16)
            print(f"  {key}: shape {diff_vectors[key].shape}")


    save_data = {
        "prompt_info": prompt_data,
        "model_name": args.model,
        "method": "last_token_two_separate_prompts",
        "analysis_timestamp": datetime.datetime.now().isoformat(),
        "generation_1": {
            "role_pair": f"{sender_1}→{receiver_1}",
            "messages": full_messages_1,
            "result": {
                "residuals": {
                    k: v.unsqueeze(0) for k, v in res_1.items() if v is not None
                },
                "generated_text": response1,
            },
        },
        "generation_2": {
            "role_pair": f"{sender_1}→{receiver_2}",
            "messages": full_messages_2,
            "result": {
                "residuals": {
                    k: v.unsqueeze(0) for k, v in res_2.items() if v is not None
                },
                "generated_text": response2,
            },
        },
        "difference_vectors": diff_vectors,
        "layer_note": args.layer,
    }
    out_name = f"{args.model}_argmax_last_token_{args.prompt_index}_従業員_to_上司.pt"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    torch.save(save_data, out_path)
    print(f"💾 已保存: {out_path}")
    print("  可与 script4 共用 load_feature_vector 加载（generation_1/2.result.residuals 格式一致）。")


if __name__ == "__main__":
    main()
