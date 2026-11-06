#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import json
import argparse
import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


_script_dir = os.path.dirname(os.path.abspath(__file__))
_script_name = os.path.splitext(os.path.basename(__file__))[0]
DETAILED_RESIDUALS_OUTPUT_DIR = os.path.join(_script_dir, _script_name)

os.makedirs(DETAILED_RESIDUALS_OUTPUT_DIR, exist_ok=True)


MODEL_PATHS = {
    "llama3": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "qwen2.5": "Qwen/Qwen2.5-7B-Instruct",
    "gemma2": "google/gemma-2-9b-it",
}


def create_emotion_prompt(emotion_type="positive", scenario="office_worker"):
    scenarios = {
        "office_worker": {
            "en": {
                "system": "You are a human office worker sharing your daily experiences. Respond naturally as a real person would.",
                "positive": (
                    "As an office worker, describe how your workday went today in a positive tone:\n"
                    "Mood: <Describe how you felt during the day in one sentence, max 30 words.>\n"
                    "Reason: <Explain what made your day good in one sentence, max 30 words.>\n"
                    "Note: Use positive vocabulary and enthusiastic tone."
                ),
                "negative": (
                    "As an office worker, describe how your workday went today in a negative tone:\n"
                    "Mood: <Describe how you felt during the day in one sentence, max 30 words.>\n"
                    "Reason: <Explain what made your day difficult in one sentence, max 30 words.>\n"
                    "Note: Use negative vocabulary and frustrated tone."
                )
            }
        },
        "student": {
            "en": {
                "system": "You are a university student sharing your exam experiences. Respond naturally as a real person would.",
                "positive": (
                    "As a student, describe how your exam went today in a positive tone:\n"
                    "Mood: <Describe how you felt about the exam in one sentence, max 30 words.>\n"
                    "Reason: <Explain what made the exam go well in one sentence, max 30 words.>\n"
                    "Note: Use positive vocabulary and confident tone."
                ),
                "negative": (
                    "As a student, describe how your exam went today in a negative tone:\n"
                    "Mood: <Describe how you felt about the exam in one sentence, max 30 words.>\n"
                    "Reason: <Explain what made the exam difficult in one sentence, max 30 words.>\n"
                    "Note: Use negative vocabulary and anxious tone."
                )
            }
        },
        "tourist": {
            "en": {
                "system": "You are a tourist sharing your travel experiences. Respond naturally as a real person would.",
                "positive": (
                    "As a tourist, describe your travel experience today in a positive tone:\n"
                    "Mood: <Describe how you felt about your trip in one sentence, max 30 words.>\n"
                    "Reason: <Explain what made your trip enjoyable in one sentence, max 30 words.>\n"
                    "Note: Use positive vocabulary and excited tone."
                ),
                "negative": (
                    "As a tourist, describe your travel experience today in a negative tone:\n"
                    "Mood: <Describe how you felt about your trip in one sentence, max 30 words.>\n"
                    "Reason: <Explain what made your trip disappointing in one sentence, max 30 words.>\n"
                    "Note: Use negative vocabulary and disappointed tone."
                )
            }
        },
        "propose": {
            "en": {
                "system": "You are a person sharing your propose experiences. Respond naturally as a real person would.",
                "positive": (
                    "As a person, describe your propose experience today in a positive tone:\n"
                    "Mood: <Describe how you felt about your propose in one sentence, max 30 words.>\n"
                    "Reason: <Explain what made your propose successful in one sentence, max 30 words.>\n"
                    "Note: Use positive vocabulary and satisfied tone."
                ),
                "negative": (
                    "As a person, describe your propose experience today in a negative tone:\n"
                    "Mood: <Describe how you felt about your propose in one sentence, max 30 words.>\n"
                    "Reason: <Explain what made your propose unsuccessful in one sentence, max 30 words.>\n"
                    "Note: Use negative vocabulary and irritated tone."
                )
            }
        },
        "chef": {
            "en": {
                "system": "You are a chef sharing your cooking experiences. Respond naturally as a real person would.",
                "positive": (
                    "As a chef, describe your cooking experience today in a positive tone:\n"
                    "Mood: <Describe how you felt about your cooking in one sentence, max 30 words.>\n"
                    "Reason: <Explain what made your cooking successful in one sentence, max 30 words.>\n"
                    "Note: Use positive vocabulary and proud tone."
                ),
                "negative": (
                    "As a chef, describe your cooking experience today in a negative tone:\n"
                    "Mood: <Describe how you felt about your cooking in one sentence, max 30 words.>\n"
                    "Reason: <Explain what made your cooking challenging in one sentence, max 30 words.>\n"
                    "Note: Use negative vocabulary and stressed tone."
                )
            }
        }
    }

    if scenario not in scenarios:
        raise ValueError(f"Unsupported scenario: {scenario}. Use one of: {list(scenarios.keys())}")

    scenario_data = scenarios[scenario]["en"]
    system_prompt = scenario_data["system"]
    user_prompt = scenario_data[emotion_type]

    return system_prompt, user_prompt


def create_emotion_conversion_prompt(target_emotion="negative", scenario="office_worker"):
    conversion_prompts = {
        "office_worker": {
            "en": {
                "negative": "Now, as the same office worker, retell your workday experience but in a negative tone. Keep the same basic facts but express them with frustration and disappointment.",
                "positive": "Now, as the same office worker, retell your workday experience but in a positive tone. Keep the same basic facts but express them with enthusiasm and satisfaction."
            }
        },
        "student": {
            "en": {
                "negative": "Now, as the same student, retell your exam experience but in a negative tone. Keep the same basic facts but express them with anxiety and stress.",
                "positive": "Now, as the same student, retell your exam experience but in a positive tone. Keep the same basic facts but express them with confidence and relief."
            }
        },
        "tourist": {
            "en": {
                "negative": "Now, as the same tourist, retell your travel experience but in a negative tone. Keep the same basic facts but express them with disappointment and dissatisfaction.",
                "positive": "Now, as the same tourist, retell your travel experience but in a positive tone. Keep the same basic facts but express them with excitement and joy."
            }
        },
        "propose": {
            "en": {
                "negative": "Now, as the same person, retell your propose experience but in a negative tone. Keep the same basic facts but express them with irritation and frustration.",
                "positive": "Now, as the same person, retell your propose experience but in a positive tone. Keep the same basic facts but express them with satisfaction and happiness."
            }
        },
        "chef": {
            "en": {
                "negative": "Now, as the same chef, retell your cooking experience but in a negative tone. Keep the same basic facts but express them with stress and disappointment.",
                "positive": "Now, as the same chef, retell your cooking experience but in a positive tone. Keep the same basic facts but express them with pride and satisfaction."
            }
        }
    }

    if scenario not in conversion_prompts:
        raise ValueError(f"Unsupported scenario: {scenario}. Use one of: {list(conversion_prompts.keys())}")

    return conversion_prompts[scenario]["en"][target_emotion]


def token_wise_generation_with_residuals(
    model, tokenizer, input_ids, past_key_values=None,
    max_new_tokens=200, sampling_method='argmax', top_p=0.9, temperature=0.7,
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
            if not capture_enabled[0]:
                return
            tensor_to_save = inputs[0] if cache_key in ["resid_pre", "resid_mid"]\
                             else (output[0] if isinstance(output, tuple) else output)
            current_step_cache[cache_key][layer_idx] = (
                tensor_to_save[:, -1, :].detach().to("cpu", copy=True).bfloat16()
            )
        return hook

    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.input_layernorm.register_forward_hook(hook_factory("resid_pre", i)))
        hooks.append(layer.post_attention_layernorm.register_forward_hook(hook_factory("resid_mid", i)))
        hooks.append(layer.register_forward_hook(hook_factory("resid_post", i)))

    with torch.no_grad():
        out = model(
            input_ids,
            use_cache=True,
            return_dict=True,
            past_key_values=past_key_values
        )
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
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            probs.scatter_(1, indices_to_remove, 0)
            next_id = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_id = torch.argmax(logits, dim=-1)

        tok_id = next_id.item()
        if tok_id == stop_token_id:
            break

        generated_token_ids.append(tok_id)
        last_tok = next_id.unsqueeze(0)

        for key in all_step_residuals.keys():
            valid_tensors = [t for t in current_step_cache[key] if t is not None]
            if len(valid_tensors) == num_layers:
                all_step_residuals[key].append(torch.stack(valid_tensors).squeeze(1))
            else:
                print(f"⚠️ 警告: 步骤 {step} 的 {key} 捕获层数 {len(valid_tensors)}/{num_layers}")


    if force_append_eot_id:
        if not generated_token_ids or generated_token_ids[-1] != force_append_eot_id:
            generated_token_ids.append(force_append_eot_id)
            last_tok = torch.tensor([[force_append_eot_id]], device=device)
            for key in current_step_cache:
                current_step_cache[key] = [None] * num_layers
            capture_enabled[0] = True
            with torch.no_grad():
                outputs = model(last_tok, use_cache=True, past_key_values=past, return_dict=True)
                past = outputs.past_key_values
            capture_enabled[0] = False
            for key in all_step_residuals.keys():
                valid_tensors = [t for t in current_step_cache[key] if t is not None]
                if valid_tensors:
                    all_step_residuals[key].append(torch.stack(valid_tensors).squeeze(1))

    for h in hooks:
        h.remove()

    final_residuals = {k: torch.stack(v) for k, v in all_step_residuals.items() if v}
    generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    return {
        "generated_text": generated_text,
        "generated_token_ids": generated_token_ids,
        "residuals": final_residuals,
        "final_past_key_values": past,
    }


def run_emotion_style_analysis(emotion_type="positive",
                             sampling_method='argmax', top_p=0.9, temperature=0.7,
                             max_samples=1, scenarios=None, no_save=False, model=None):
    if scenarios is None:
        scenarios = ["office_worker", "student", "tourist", "propose", "chef"]

    if model is not None:
        if model not in MODEL_PATHS:
            raise ValueError(f"Unsupported model: {model}. Use one of {list(MODEL_PATHS.keys())}")
        model_name = model
        model_path = MODEL_PATHS[model]
    else:
        model_name = "llama3"
        model_path = MODEL_PATHS["llama3"]


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"🚀 加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    eot_id = tokenizer.eos_token_id


    target_emotion = "negative" if emotion_type == "positive" else "positive"


    total_tasks = len(scenarios) * max_samples
    current_task = 0

    for scenario in scenarios:
        print(f"\n{'='*80}")
        print(f"🎬 场景: {scenario}")
        print(f"{'='*80}")


        for sample_idx in range(max_samples):
            current_task += 1
            print(f"\n{'='*80}")
            print(f"📋 总体进度: {current_task}/{total_tasks}")
            print(f"🎯 场景: {scenario} | 样本: {sample_idx + 1}/{max_samples}")
            print(f"🎯 模型: {model_name} | 初始情绪: {emotion_type} → 目标情绪: {target_emotion}")

            try:
                print("--- 阶段 1: 正在生成初始情绪内容... ---")


                system_prompt, user_prompt_1 = create_emotion_prompt(
                    emotion_type=emotion_type, scenario=scenario
                )

                if model_name == "gemma2":
                    messages_1 = [
                        {"role": "user", "content": f"{system_prompt}\n\n{user_prompt_1}"}
                    ]
                else:
                    messages_1 = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt_1}
                    ]

                prompt_text_1 = tokenizer.apply_chat_template(messages_1, tokenize=False, add_generation_prompt=True)
                input_ids_1 = tokenizer(prompt_text_1, return_tensors="pt").to(device)["input_ids"]

                gen_result_1 = token_wise_generation_with_residuals(
                    model, tokenizer, input_ids=input_ids_1, max_new_tokens=80,
                    sampling_method=sampling_method, top_p=top_p, temperature=temperature,
                    force_append_eot_id=eot_id
                )
                print(f"✅ 阶段 1 完成. 生成了 {len(gen_result_1['generated_token_ids'])} 个 token.")
                print(f"   生成内容: {gen_result_1['generated_text'][:100]}...")

                print("--- 阶段 2: 正在进行情绪风格转换... ---")


                conversion_prompt = create_emotion_conversion_prompt(
                    target_emotion=target_emotion, scenario=scenario
                )

                if model_name == "gemma2":
                    messages_for_suffix = [
                        {"role": "user", "content": conversion_prompt}
                    ]
                else:
                    messages_for_suffix = [
                        {"role": "assistant", "content": gen_result_1["generated_text"]},
                        {"role": "user", "content": conversion_prompt}
                    ]

                suffix_text = tokenizer.apply_chat_template(messages_for_suffix, tokenize=False, add_generation_prompt=True)
                suffix_ids = tokenizer(suffix_text, return_tensors="pt", add_special_tokens=False).to(device)["input_ids"]

                print(f"  KV缓存增量计算: 后缀长度 {suffix_ids.shape[1]}")

                gen_result_2 = token_wise_generation_with_residuals(
                    model, tokenizer, input_ids=suffix_ids,
                    past_key_values=gen_result_1["final_past_key_values"],
                    max_new_tokens=80, sampling_method=sampling_method,
                    top_p=top_p, temperature=temperature, force_append_eot_id=eot_id
                )
                print(f"✅ 阶段 2 完成. 生成了 {len(gen_result_2['generated_token_ids'])} 个 token.")
                print(f"   生成内容: {gen_result_2['generated_text'][:100]}...")


                output_file = os.path.join(
                    DETAILED_RESIDUALS_OUTPUT_DIR,
                    f"emotion_style_{model_name}_en_{scenario}_{emotion_type}_{sampling_method}_sample_{sample_idx}.pt"
                )


                del gen_result_1["final_past_key_values"]
                if "final_past_key_values" in gen_result_2:
                    del gen_result_2["final_past_key_values"]


                if model_name == "gemma2":
                    full_messages_2 = messages_1 + [
                        {"role": "assistant", "content": gen_result_1["generated_text"]},
                        {"role": "user", "content": conversion_prompt}
                    ]
                else:
                    full_messages_2 = messages_1 + messages_for_suffix

                save_data = {
                    "sample_info": {
                        "sample_idx": sample_idx,
                        "scenario": scenario,
                        "model_name": model_name,
                        "initial_emotion": emotion_type,
                        "target_emotion": target_emotion
                    },
                    "model_name": model_name,
                    "analysis_timestamp": datetime.datetime.now().isoformat(),
                    "sampling_config": {
                        "method": sampling_method,
                        "top_p": top_p,
                        "temperature": temperature
                    },
                    "generation_1": {
                        "scenario": scenario,
                        "emotion_type": emotion_type,
                        "messages": messages_1,
                        "result": gen_result_1
                    },
                    "generation_2": {
                        "scenario": scenario,
                        "emotion_type": target_emotion,
                        "messages": full_messages_2,
                        "result": gen_result_2
                    }
                }

                if not no_save:
                    torch.save(save_data, output_file)
                    print(f"💾 已保存详细残差数据到: {output_file}")
                else:
                    print("⚠️ 按照配置跳过保存文件（no_save=True）")

                torch.cuda.empty_cache()

            except Exception as e:
                print(f"⚠️ 处理场景 {scenario} 样本 {sample_idx} 失败: {e}")
                import traceback
                traceback.print_exc()

    print(f"\n🎉 情绪风格转换分析完成！共处理 {len(scenarios)} 个场景，每个场景 {max_samples} 个样本。")


def main():
    parser = argparse.ArgumentParser(description='Extract cached.')
    parser.add_argument("--model", type=str, default=None, choices=list(MODEL_PATHS.keys()),
                        help='Model alias or Hugging Face model path.')
    parser.add_argument("--emotion_type", type=str, default="positive", choices=["positive", "negative"], help='Initial sentiment.')
    parser.add_argument("--sampling_method", type=str, default="argmax", choices=["argmax", "sample"])
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_samples", type=int, default=1, help='Extraction pairs per scenario.')
    parser.add_argument("--scenarios", type=str, nargs='+', default=None,
                       choices=["office_worker", "student", "tourist", "propose", "chef", "all"],
                       help='Scenario names.')
    parser.add_argument("--no_save", action="store_true",
                       help='Skip writing captured residuals.')
    args = parser.parse_args()

    scenarios = args.scenarios
    if scenarios is None or "all" in scenarios:
        scenarios = ["office_worker", "student", "tourist", "propose", "chef"]

    print(f"🎬 将运行以下场景: {', '.join(scenarios)}")
    print(f"📊 每个场景生成 {args.max_samples} 个样本")
    print(f"📦 总共将生成 {len(scenarios) * args.max_samples} 个数据文件")
    if args.model:
        print(f"🤖 指定模型: {args.model}")

    run_emotion_style_analysis(
        emotion_type=args.emotion_type,
        sampling_method=args.sampling_method,
        top_p=args.top_p,
        temperature=args.temperature,
        max_samples=args.max_samples,
        scenarios=scenarios,
        no_save=args.no_save,
        model=args.model
    )

if __name__ == "__main__":
    main()
