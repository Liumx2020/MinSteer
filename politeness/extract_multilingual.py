#!/usr/bin/env python
# -*- coding: utf-8 -*-


from release_paths import project_path
import os
import sys
import argparse
import datetime
import importlib.util
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE_DIR = project_path('politeness')
OUTPUT_DIR = os.path.join(BASE_DIR, "vectors_english")
os.makedirs(OUTPUT_DIR, exist_ok=True)


SCRIPT3 = os.path.join(BASE_DIR, "extract_cached.py")
spec = importlib.util.spec_from_file_location("residual_extract", SCRIPT3)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
token_wise_generation_with_residuals = mod.token_wise_generation_with_residuals

FORMAL_TEMPLATE = "Rewrite the following request to be MORE FORMAL and polite. Output only the rewritten text, 1-3 sentences:\n\n{text}"
CASUAL_TEMPLATE = "Rewrite the following request to be MORE CASUAL and informal. Output only the rewritten text, 1-3 sentences:\n\n{text}"

BUILTIN_TEXTS = [
    "Can you explain more about how this works?",
    "I need help with the deadline. When can we discuss it?",
    "Could you please send me the document when you have time?",
    "What is the best way to approach this problem?",
    "I would like to request a meeting to go over the details.",
]


CN_FORMAL_TEMPLATE = "请将以下句子改写得更正式、更有礼貌。只输出改写结果，不要解释：\n\n{text}"
CN_CASUAL_TEMPLATE = "请将以下句子改写得更口语化、更随意。只输出改写结果，不要解释：\n\n{text}"

CN_BUILTIN_TEXTS = [
    "这个功能怎么用？",
    "你能帮我看看这个问题吗？",
    "代码运行太慢了怎么优化？",
    "明天能不能开个会讨论一下？",
    "请问这个错误是什么意思？",
]

CN_CORPUS_JSON = os.path.join(BASE_DIR, "chinese_requests.json")
CN_OUTPUT_DIR = os.path.join(BASE_DIR, "vectors_chinese")
os.makedirs(CN_OUTPUT_DIR, exist_ok=True)


def get_chinese_prompts(max_prompts=10, use_corpus_json=True):
    if use_corpus_json and os.path.exists(CN_CORPUS_JSON):
        try:
            import json
            with open(CN_CORPUS_JSON, encoding="utf-8") as f:
                data = json.load(f)
            items = data.get("items", [])
            neutral = [x["text"] for x in items if x.get("orig_binary") == 0]
            if neutral:
                return neutral[:max_prompts]

            all_texts = [x["text"] for x in items[:max_prompts * 2]]
            return all_texts[:max_prompts]
        except Exception as e:
            print(f"⚠ 中文语料加载失败: {e}")
    return CN_BUILTIN_TEXTS[:max_prompts]


def get_english_prompts(max_prompts=10, use_stanford=True):
    if use_stanford:
        try:
            from convokit import Corpus, download
            corpus = Corpus(filename=download("stack-exchange-politeness-corpus"))
            texts = []
            for utt in corpus.iter_utterances():
                t = (utt.text or "").strip()
                if 30 <= len(t) <= 300 and "compile" not in t.lower()[:50]:
                    texts.append(t)
                    if len(texts) >= max_prompts:
                        break
            if texts:
                return texts
        except Exception as e:
            print(f"⚠ Stanford 加载失败: {e}")
    return BUILTIN_TEXTS[:max_prompts]


def extract_english_vector(model_name="swallow", max_prompts=5, layer=15, residual_type="resid_pre"):
    model_paths = {
        "swallow":   "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3",
        "llama_en":  "meta-llama/Llama-3.1-8B-Instruct",
        "llama2":    "meta-llama/Llama-2-7b-chat-hf",
        "llama3_zh": "shenzhi-wang/Llama3-8B-Chinese-Chat",
        "mistral":   "mistralai/Mistral-7B-Instruct-v0.3",
    }
    model_path = model_paths.get(model_name, model_paths["swallow"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"🚀 加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    eot_id = tokenizer.eos_token_id

    system = "You are rewriting requests. Keep the meaning, adjust the tone as instructed."
    texts = get_english_prompts(max_prompts)
    print(f"📂 共 {len(texts)} 条英语文本")

    all_formal_resid = {k: [] for k in ["resid_pre", "resid_mid", "resid_post"]}
    all_casual_resid = {k: [] for k in all_formal_resid}

    for i, text in enumerate(texts):
        print(f"\n--- 处理 {i+1}/{len(texts)} ---")
        for style, template in [("formal", FORMAL_TEMPLATE), ("casual", CASUAL_TEMPLATE)]:
            user = template.format(text=text)
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            input_ids = tokenizer(prompt_text, return_tensors="pt").to(device)["input_ids"]
            try:
                out = token_wise_generation_with_residuals(
                    model, tokenizer, input_ids=input_ids, max_new_tokens=150,
                    sampling_method="argmax", force_append_eot_id=eot_id,
                )
                for k in all_formal_resid:
                    r = out["residuals"].get(k)
                    if r is not None:
                        (all_formal_resid if style == "formal" else all_casual_resid)[k].append(r)
                print(f"  {style}: {out['generated_text'][:80]}...")
            except Exception as e:
                print(f"  ⚠ {style} 失败: {e}")
        torch.cuda.empty_cache()


    result = {}
    n_pairs = min(len(all_formal_resid["resid_pre"]), len(all_casual_resid["resid_pre"]))
    for k in all_formal_resid:
        if n_pairs == 0:
            continue
        diffs = []
        for i in range(n_pairs):
            f = all_formal_resid[k][i]
            c = all_casual_resid[k][i]
            min_len = min(f.shape[0], c.shape[0])
            avg_f = f[:min_len].float().mean(dim=0)
            avg_c = c[:min_len].float().mean(dim=0)
            diffs.append(avg_c - avg_f)
        vec = torch.stack(diffs).mean(dim=0)
        result[k] = vec
        print(f"  {k}: shape {vec.shape} (n_pairs={n_pairs})")

    out_file = os.path.join(OUTPUT_DIR, f"english_formal_casual_{model_name}_layer{layer}.pt")
    save_data = {
        "model_name": model_name,
        "model_path": model_path,
        "direction": "casual_minus_formal",
        "layer": layer,
        "residual_type": residual_type,
        "vectors": result,
        "n_prompts": len(texts),
        "timestamp": datetime.datetime.now().isoformat(),
    }
    torch.save(save_data, out_file)
    print(f"\n💾 已保存: {out_file}")
    return out_file


def extract_chinese_vector(model_name="llama3_zh", max_prompts=5, layer=15, residual_type="resid_pre"):
    model_paths = {
        "llama3_zh": "shenzhi-wang/Llama3-8B-Chinese-Chat",
    }
    model_path = model_paths.get(model_name, model_paths["llama3_zh"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"🚀 加载模型(中文): {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    eot_id = tokenizer.eos_token_id

    system = "你是中文改写助手。保持原句意思，按指令调整语气。只输出改写结果。"
    texts = get_chinese_prompts(max_prompts)
    print(f"📂 共 {len(texts)} 条中文文本")

    all_formal_resid = {k: [] for k in ["resid_pre", "resid_mid", "resid_post"]}
    all_casual_resid = {k: [] for k in all_formal_resid}

    for i, text in enumerate(texts):
        print(f"\n--- 处理 {i+1}/{len(texts)} ---")
        for style, template in [("formal", CN_FORMAL_TEMPLATE), ("casual", CN_CASUAL_TEMPLATE)]:
            user = template.format(text=text)
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            input_ids = tokenizer(prompt_text, return_tensors="pt").to(device)["input_ids"]
            try:
                out = token_wise_generation_with_residuals(
                    model, tokenizer, input_ids=input_ids, max_new_tokens=150,
                    sampling_method="argmax", force_append_eot_id=eot_id,
                )
                for k in all_formal_resid:
                    r = out["residuals"].get(k)
                    if r is not None:
                        (all_formal_resid if style == "formal" else all_casual_resid)[k].append(r)
                print(f"  {style}: {out['generated_text'][:80]}...")
            except Exception as e:
                print(f"  ⚠ {style} 失败: {e}")
        torch.cuda.empty_cache()

    result = {}
    n_pairs = min(len(all_formal_resid["resid_pre"]), len(all_casual_resid["resid_pre"]))
    for k in all_formal_resid:
        if n_pairs == 0:
            continue
        diffs = []
        for i in range(n_pairs):
            f = all_formal_resid[k][i]
            c = all_casual_resid[k][i]
            min_len = min(f.shape[0], c.shape[0])
            avg_f = f[:min_len].float().mean(dim=0)
            avg_c = c[:min_len].float().mean(dim=0)
            diffs.append(avg_c - avg_f)
        vec = torch.stack(diffs).mean(dim=0)
        result[k] = vec
        print(f"  {k}: shape {vec.shape} (n_pairs={n_pairs})")

    out_file = os.path.join(CN_OUTPUT_DIR, f"chinese_formal_casual_{model_name}_layer{layer}.pt")
    save_data = {
        "model_name": model_name,
        "model_path": model_path,
        "direction": "casual_minus_formal",
        "layer": layer,
        "residual_type": residual_type,
        "vectors": result,
        "n_prompts": len(texts),
        "language": "chinese",
        "timestamp": datetime.datetime.now().isoformat(),
    }
    torch.save(save_data, out_file)
    print(f"\n💾 已保存: {out_file}")
    return out_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, default="en", choices=["en", "cn"],
                        help='Language.')
    parser.add_argument("--model", type=str, default="swallow",
                        choices=["swallow", "llama_en", "llama2", "llama3_zh", "mistral"])
    parser.add_argument("--max_prompts", type=int, default=5)
    parser.add_argument("--layer", type=int, default=15)
    args = parser.parse_args()
    if args.language == "cn":
        if args.model != "llama3_zh":
            args.model = "llama3_zh"
            print("中文向量仅支持 llama3_zh，已自动切换")
        extract_chinese_vector(args.model, args.max_prompts, args.layer)
    else:
        extract_english_vector(args.model, args.max_prompts, args.layer)


if __name__ == "__main__":
    main()
