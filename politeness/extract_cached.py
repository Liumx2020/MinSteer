#!/usr/bin/env python
# -*- coding: utf-8 -*-


from release_paths import project_path
import os
import json
import argparse
import torch
import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM


BASE_DIR = project_path('politeness')
SCRIPT_FOLDER = os.path.join(BASE_DIR, "extract_cached")
SHORT_EMAILS_INPUT = project_path('prompts/politeness_scenarios.json')
DETAILED_RESIDUALS_OUTPUT_DIR = SCRIPT_FOLDER

if not os.path.exists(DETAILED_RESIDUALS_OUTPUT_DIR):
    os.makedirs(DETAILED_RESIDUALS_OUTPUT_DIR)


def create_initial_user_prompt(scenario, sender_identity, receiver_identity):
    role_guidance = ""
    if sender_identity == "教員" and receiver_identity == "学生":
        role_guidance = """
【重要】あなたは教員として学生にメールを書いています。以下の点に注意してください：
- 教員として適度な権威性を保ちつつ、指導的な語調を使用する
- 学生に対しては丁寧語を使うが、過度に謙譲語を使わない
- 「〜してください」「〜しなさい」など、指示・依頼の表现を適切に使用する
- 教育的配慮を示しつつも、上下関係を適切に表現する
"""
    elif sender_identity == "学生" and receiver_identity == "教授":
        role_guidance = """
【重要】あなたは学生として教授にメールを書いています。以下の点に注意してください：
- 教授に対する深い敬意を示し、謙譲語と尊敬語を適切に使用する
- 「お忙しい中恐れ入ります」「ご指導いただき」などの謙譲表現を使用する
- 自分の立場を謙虚に表現し、相手の時間を尊重する姿勢を示す
- 丁寧で恭しい文体を維持する
"""
    elif sender_identity == "学生" and receiver_identity == "友人":
        role_guidance = """
【重要】あなたは学生として友人にメールを書いています。以下の点に注意してください：

【自然な友人間のコミュニケーション】
- 親しみやすく自然な語調を使用する
- 過度な敬語は避け、丁寧語程度に留める
- 「させていただきます」のような謙譲語は友人には不自然
- 「します」「行います」など、シンプルな表現を使用

【友人らしい表現】
- 「お疲れ様」または軽い挨拶で開始
- 「～だと思う」「～しよう」「～はどう？」など親近感のある表現
- 「よろしく」または「また連絡するね」で自然に終了
- 同等の立場として気軽にコミュニケーションを取る

【注意】友人同士の自然な日本語を心がけ、堅すぎない表現を使用してください。
"""
    elif sender_identity == "従業員" and receiver_identity == "部下":
        role_guidance = """
【重要】あなたは上司として部下にメールを書いています。以下の点に注意してください：

🚫【絶対禁止表現】以下を使用した場合は失敗です：
- 「させていただきます」→「します」のみ使用
- 「確認させていただきます」→「確認します」
- 「対応させていただきます」→「対応します」
- 「検討させていただきます」→「検討します」
- 「よろしくお願いいたします」→「よろしく。」

✅【必須使用表現】
- 「お疲れ様です」で開始
- 「確認します」「対応します」「検討します」
- 「～してください」で指示
- 「よろしく。」で終了

【例】
❌ 間違い：「打ち合わせ後、確認させていただきます。」
✅ 正解：「打ち合わせ後、確認します。」

❌ 間違い：「よろしくお願いいたします。」
✅ 正解：「よろしく。」

【重要】上司が部下に「させていただきます」は不自然です。
自然で適切な上司の話し方を使用してください。
"""
    elif sender_identity == "従業員" and receiver_identity == "上司":
        role_guidance = """
【重要】あなたは部下として上司にメールを書いています。以下の点に注意してください：
- 上司に対する敬意を示し、謙譲語と尊敬語を適切に使用する
- 「お疲れ様です」「恐れ入ります」などの謙譲表現を使用する
- 報告・相談の際は簡潔で分かりやすい構成にする
- 相手の時間を尊重し、丁寧で恭しい文体を維持する
"""
    elif sender_identity == "従業員" and receiver_identity == "同僚":
        role_guidance = """
【重要】あなたは同僚として同じ立場の人にメールを書いています。以下の点に注意してください：
- 同等の立場として適度な距離感を保つ
- 丁寧語を基本とするが、過度な謙譲語は避ける
- 協力的で建設的な語調を使用する
- 「お疲れ様です」「よろしくお願いします」など、ビジネス的な礼儀を保つ
"""
    else:
        role_guidance = f"""
【重要】あなたは{sender_identity}として{receiver_identity}にメールを書いています。
相手との関係性を考慮し、適切な敬語レベルと語調を選択してください。
"""

    format_instruction = """
【出力形式】以下の形式で必ずメールを作成してください：
件名：[適切な件名]
宛先：[相手の呼び方（○○さん、○○様など）]
本文：[メールの本文内容]
"""

    full_user_prompt = f"{role_guidance}\n{format_instruction}\n\n【具体的な指示】\n{scenario}"
    return full_user_prompt

def token_wise_generation_with_residuals(
    model, tokenizer,
    input_ids,
    past_key_values=None,
    max_new_tokens=200,
    sampling_method='argmax',
    top_p=0.9,
    temperature=0.7,
    force_append_eot_id=None
):
    device = next(model.parameters()).device
    num_layers = model.config.num_hidden_layers


    capture_enabled = [False]

    hooks = []
    all_step_residuals = {"resid_pre": [], "resid_mid": [], "resid_post": []}


    current_step_cache = {
        "resid_pre": [None] * num_layers,
        "resid_mid": [None] * num_layers,
        "resid_post": [None] * num_layers
    }

    def hook_factory(cache_key, layer_idx):
        def hook(module, inputs, output):
            if not capture_enabled[0]: return

            tensor_to_save = inputs[0] if cache_key in ["resid_pre", "resid_mid"] else (output[0] if isinstance(output, tuple) else output)

            current_step_cache[cache_key][layer_idx] = tensor_to_save[:, -1, :].detach().to("cpu", copy=True).bfloat16()
        return hook


    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.input_layernorm.register_forward_hook(hook_factory("resid_pre", i)))
        hooks.append(layer.post_attention_layernorm.register_forward_hook(hook_factory("resid_mid", i)))
        hooks.append(layer.register_forward_hook(hook_factory("resid_post", i)))


    with torch.no_grad():
        if past_key_values is None:
            out = model(input_ids, use_cache=True, return_dict=True)
        else:
            out = model(input_ids, use_cache=True, past_key_values=past_key_values, return_dict=True)
        past = out.past_key_values
        last_tok = input_ids[:, -1:]

    generated_token_ids = []
    stop_token_id = tokenizer.eos_token_id


    for step in range(max_new_tokens):

        for key in current_step_cache:
            current_step_cache[key] = [None] * num_layers

        capture_enabled[0] = True
        with torch.no_grad():
            outputs = model(last_tok, use_cache=True, past_key_values=past, return_dict=True)
            logits = outputs.logits[:, -1, :]
            past = outputs.past_key_values
        capture_enabled[0] = False

        if sampling_method == 'sample':

            scaled_logits = logits / temperature
            probs = torch.softmax(scaled_logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone(); sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            probs.scatter_(1, indices_to_remove, 0)
            next_id = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_id = torch.argmax(logits, dim=-1)

        tok_id = next_id.item()
        if tok_id == stop_token_id: break

        generated_token_ids.append(tok_id)
        last_tok = next_id.unsqueeze(0)


        for key in all_step_residuals.keys():

            valid_tensors = [t for t in current_step_cache[key] if t is not None]
            if len(valid_tensors) == num_layers:
                 all_step_residuals[key].append(torch.stack(valid_tensors).squeeze(1))
            else:
                print(f"⚠️ 警告: 在步骤 {step} 的 {key} 中，期望捕获 {num_layers} 层，实际捕获 {len(valid_tensors)} 层。")


    if force_append_eot_id:
        if not generated_token_ids or generated_token_ids[-1] != force_append_eot_id:
            generated_token_ids.append(force_append_eot_id)
            last_tok = torch.tensor([[force_append_eot_id]], device=device)

            for key in current_step_cache: current_step_cache[key] = [None] * num_layers
            capture_enabled[0] = True
            with torch.no_grad():
                outputs = model(last_tok, use_cache=True, past_key_values=past, return_dict=True)
                past = outputs.past_key_values
            capture_enabled[0] = False

            for key in all_step_residuals.keys():
                valid_tensors = [t for t in current_step_cache[key] if t is not None]
                if valid_tensors: all_step_residuals[key].append(torch.stack(valid_tensors).squeeze(1))

    for h in hooks: h.remove()


    final_residuals = {key: torch.stack(res_list) for key, res_list in all_step_residuals.items() if res_list}
    generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)

    return {
        "generated_text": generated_text,
        "generated_token_ids": generated_token_ids,
        "residuals": final_residuals,
        "final_past_key_values": past,
    }


def detailed_residual_analysis(model_name="swallow", max_prompts=None, sampling_method='argmax', top_p=0.9, temperature=0.7):
    try:
        with open(SHORT_EMAILS_INPUT, 'r', encoding='utf-8') as f: short_emails_data = json.load(f)
    except FileNotFoundError: print(f"❌ 找不到文件: {SHORT_EMAILS_INPUT}"); return

    prompts_for_analysis = short_emails_data["prompts_for_detailed_analysis"]

    prompts_for_analysis = [prompt for prompt in prompts_for_analysis if prompt.get('role_pair') == "従業員→上司"]
    if max_prompts: prompts_for_analysis = prompts_for_analysis[:max_prompts]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3" if model_name == "swallow" else "elyza/Llama-3-ELYZA-JP-8B"

    print(f"🚀 加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)

    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    eot_id = tokenizer.eos_token_id

    role_transition_map = {"教授": "友人", "上司": "友人", "部下": "同僚", "学生": "友人", "友人": "教授", "同僚": "上司"}
    SYSTEM_PROMPT = "あなたは日本人のメール作成アシスタントです。以下の指示に従って、適切なメールを一通だけ書いてください。"

    for prompt_idx, prompt_data in enumerate(prompts_for_analysis):
        print(f"\n{'='*80}")
        print(f"📋 处理进度: {prompt_idx + 1}/{len(prompts_for_analysis)}")
        print(f"🎯 场景: {prompt_data['role_pair']} | 采样策略: {sampling_method}")

        try:
            print("--- 阶段 1: 正在生成初始邮件... ---")
            sender_1, receiver_1 = prompt_data["sender"], prompt_data["receiver"]

            user_prompt_1 = create_initial_user_prompt(prompt_data['scenario'], sender_1, receiver_1)
            messages_1 = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt_1},
            ]

            prompt_text_1 = tokenizer.apply_chat_template(messages_1, tokenize=False, add_generation_prompt=True)
            prompt_ids_1 = tokenizer(prompt_text_1, return_tensors="pt").to(device)["input_ids"]

            gen_result_1 = token_wise_generation_with_residuals(
                model, tokenizer, input_ids=prompt_ids_1, max_new_tokens=200,
                sampling_method=sampling_method, top_p=top_p, temperature=temperature, force_append_eot_id=eot_id
            )
            print(f"✅ 阶段 1 完成. 生成了 {len(gen_result_1['generated_token_ids'])} 个 token.")

            print("--- 阶段 2: 正在根据修改指令增量生成新邮件... ---")
            receiver_2 = role_transition_map.get(receiver_1)
            if not receiver_2: print(f"⚠️ 找不到角色转换规则: {receiver_1}"); continue

            modification_prompt = f"そのメールの内容を変えずに、受信者を「{receiver_1}」から「{receiver_2}」に変更して書き直してください。"

            messages_for_suffix = [
                {"role": "assistant", "content": gen_result_1["generated_text"]},
                {"role": "user", "content": modification_prompt}
            ]

            suffix_text = tokenizer.apply_chat_template(messages_for_suffix, tokenize=False, add_generation_prompt=True)

            suffix_ids = tokenizer(suffix_text, return_tensors="pt", add_special_tokens=False).to(device)["input_ids"]

            print(f"  KV缓存增量计算: 后缀长度 {suffix_ids.shape[1]}")

            gen_result_2 = token_wise_generation_with_residuals(
                model, tokenizer, input_ids=suffix_ids, past_key_values=gen_result_1["final_past_key_values"],
                max_new_tokens=200, sampling_method=sampling_method, top_p=top_p, temperature=temperature, force_append_eot_id=eot_id
            )
            print(f"✅ 阶段 2 完成. 生成了 {len(gen_result_2['generated_token_ids'])} 个 token.")

            output_file = os.path.join(DETAILED_RESIDUALS_OUTPUT_DIR, f"{model_name}_{sampling_method}_detailed_{prompt_idx}_{prompt_data['role_pair'].replace('→', '_to_')}.pt")

            del gen_result_1["final_past_key_values"]
            if "final_past_key_values" in gen_result_2: del gen_result_2["final_past_key_values"]

            full_messages_2 = messages_1 + messages_for_suffix

            save_data = {
                "prompt_info": prompt_data, "model_name": model_name, "analysis_timestamp": datetime.datetime.now().isoformat(),
                "sampling_config": {"method": sampling_method, "top_p": top_p, "temperature": temperature},
                "generation_1": {"role_pair": f"{sender_1}→{receiver_1}", "messages": messages_1, "result": gen_result_1},
                "generation_2": {"role_pair": f"{sender_1}→{receiver_2}", "messages": full_messages_2, "result": gen_result_2}
            }
            torch.save(save_data, output_file)
            print(f"💾 已保存两阶段的详细残差数据到: {output_file}")

            torch.cuda.empty_cache()

        except Exception as e:
            print(f"⚠️ 处理prompt {prompt_idx} 失败: {e}")
            import traceback; traceback.print_exc()

    print(f"\n🎉 详细残差流分析完成！")


def main():
    parser = argparse.ArgumentParser(description='Extract cached.')
    parser.add_argument("--model", type=str, required=True, choices=["swallow", "elyza"], help='Model alias or Hugging Face model path.')
    parser.add_argument("--max_prompts", type=int, default=None, help='Maximum number of prompts.')
    parser.add_argument("--sampling_method", type=str, default="argmax", choices=['argmax', 'sample'], help='Greedy or sampled generation.')
    parser.add_argument("--top_p", type=float, default=0.9, help='Nucleus sampling threshold.')
    parser.add_argument("--temperature", type=float, default=0.7, help='Sampling temperature.')

    args = parser.parse_args()

    detailed_residual_analysis(args.model, args.max_prompts, args.sampling_method, args.top_p, args.temperature)

if __name__ == "__main__":
    main()
