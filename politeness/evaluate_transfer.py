#!/usr/bin/env python
# -*- coding: utf-8 -*-


from release_paths import project_path
import os
import sys
import argparse
import datetime
import json
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


RANDOM_SEED = 42
def _fix_random_seed(seed=RANDOM_SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


_FORMALITY_CLF = None
_FORMALITY_DEV = None

def _get_formality_clf(device):
    global _FORMALITY_CLF, _FORMALITY_DEV
    dev = 0 if (device.type == "cuda") else -1
    if _FORMALITY_CLF is None or _FORMALITY_DEV != dev:
        try:
            from transformers import pipeline
            _FORMALITY_CLF = pipeline(
                "text-classification", model="s-nlp/roberta-base-formality-ranker",
                device=dev, truncation=True, max_length=128,
            )
            _FORMALITY_DEV = dev
            print(f"[formality] RoBERTa ranker loaded (device={dev})")
        except Exception as e:
            print(f"⚠ Formality ranker 跳过: {e}")
            _FORMALITY_CLF = "skip"
    return _FORMALITY_CLF if _FORMALITY_CLF != "skip" else None

def _formality_prob(text, clf):
    if not text or not clf:
        return 0.5
    try:
        r = clf(text[:512])[0]
        lab, sc = r["label"].lower(), r["score"]
        return sc if lab == "formal" else 1.0 - sc
    except Exception:
        return 0.5


def _cn_formality_score(text):
    if not text or not text.strip():
        return 0.5
    t = text.strip()
    formal_markers = ["您", "请", "麻烦", "感谢", "敬请", "劳烦", "请问", "是否", "期待", "不知", "幸甚", "恳请", "赐教", "感激", "多谢"]
    informal_markers = ["你", "啊", "呢", "哈", "嗯", "哦", "咋", "咋回事", "咋整", "搞啥", "啥意思", "赶紧", "快点", "别废话", "磨蹭"]
    f_count = sum(t.count(m) for m in formal_markers)
    i_count = sum(t.count(m) for m in informal_markers)
    total = f_count + i_count
    if total == 0:
        return 0.5

    raw = f_count / total
    return min(1.0, max(0.0, 0.5 + (raw - 0.5) * 1.2))


BASE_DIR = project_path('politeness')
FEATURE_VECTOR_DIR = os.path.join(BASE_DIR, "extract_cached")
ENGLISH_VECTOR_DIR = os.path.join(BASE_DIR, "vectors_english")
FEATURE_VECTOR_FILENAME = "swallow_argmax_detailed_1_従業員_to_上司.pt"
OUTPUT_DIR = os.path.join(BASE_DIR, "evaluate_transfer")
os.makedirs(OUTPUT_DIR, exist_ok=True)

INTERVENTION_LAYER = 15
INTERVENTION_RESIDUAL_TYPE = "resid_pre"
MODEL_PATH = "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3"

HIDDEN_SIZE = 4096


def load_feature_vector_english(file_path, layer_idx, residual_type):
    try:
        data = torch.load(file_path, map_location="cpu")
    except FileNotFoundError:
        return None
    vec = data.get("vectors", {}).get(residual_type)
    if vec is None:
        return None
    if vec.dim() == 1:
        return vec.float()
    return vec[layer_idx, :].float()
    # vec shape: [num_layers, hidden_size] or [hidden_size]


CN_VECTOR_DIR = os.path.join(BASE_DIR, "vectors_chinese")
CN_CORPUS_JSON = os.path.join(BASE_DIR, "chinese_requests.json")


def load_feature_vector_chinese(file_path, layer_idx, residual_type):
    try:
        data = torch.load(file_path, map_location="cpu")
    except FileNotFoundError:
        return None
    vec = data.get("vectors", {}).get(residual_type)
    if vec is None:
        return None
    if vec.dim() == 1:
        return vec.float()
    return vec[layer_idx, :].float()


def load_feature_vector(file_path, layer_idx, residual_type, fake_vector="none", readout="truncated"):
    if fake_vector == "zero":
        v = torch.zeros(HIDDEN_SIZE)
        print("❷ 使用假向量: zero（对照实验）")
        return v
    if fake_vector == "random":
        v = torch.randn(HIDDEN_SIZE) / (HIDDEN_SIZE ** 0.5)
        print("❷ 使用假向量: random（对照实验）")
        return v
    if readout == 'full_mean':
        from politeness.geometry import load_vector_all_layers
        return load_vector_all_layers(file_path, residual_type)[layer_idx]
    if readout != 'truncated':
        raise ValueError(f'Unknown vector readout: {readout}')
    print(f"❷ 加载敬语向量: {file_path}")
    try:
        data = torch.load(file_path, map_location="cpu")
    except FileNotFoundError:
        print(f"❌ 文件不存在，请先运行脚本3: {file_path}")
        return None
    residuals_g1 = data["generation_1"]["result"]["residuals"][residual_type]
    residuals_g2 = data["generation_2"]["result"]["residuals"][residual_type]
    layer_residuals_g1 = residuals_g1[:, layer_idx, :]
    layer_residuals_g2 = residuals_g2[:, layer_idx, :]
    min_len = min(layer_residuals_g1.shape[0], layer_residuals_g2.shape[0])
    avg_g1 = layer_residuals_g1[:min_len, :].mean(dim=0)
    avg_g2 = layer_residuals_g2[:min_len, :].mean(dim=0)
    feature_vector = avg_g2 - avg_g1
    print("向量加载完成 (友人 - 上司)")
    return feature_vector


def generate_with_intervention(model, tokenizer, input_ids, attention_mask=None, intervention_vector=None,
                               intervention_layer=None, intervention_strength=0.0, residual_type="resid_pre"):
    device = next(model.parameters()).device
    hooks = []
    if intervention_vector is not None and intervention_layer is not None and intervention_strength != 0:
        def intervention_hook(module, inputs, output):
            inputs[0][:, -1, :] += intervention_strength * intervention_vector.to(device)
            return inputs[0]
        target_layer = model.model.layers[intervention_layer]
        if residual_type == "resid_pre":
            hook = target_layer.input_layernorm.register_forward_pre_hook(
                lambda m, i: intervention_hook(m, i, None))
        elif residual_type == "resid_mid":
            hook = target_layer.post_attention_layernorm.register_forward_pre_hook(
                lambda m, i: intervention_hook(m, i, None))
        else:
            def post_hook(module, inputs, output):
                output[0][:, -1, :] += intervention_strength * intervention_vector.to(device)
                return output
            hook = target_layer.register_forward_hook(post_hook)
        hooks.append(hook)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    gen_kw = {"max_new_tokens": 150, "use_cache": True, "do_sample": False,
              "eos_token_id": tokenizer.eos_token_id, "pad_token_id": pad_id}
    if attention_mask is not None:
        gen_kw["attention_mask"] = attention_mask
    with torch.no_grad():
        output_ids = model.generate(input_ids, **gen_kw)
    for h in hooks:
        h.remove()
    return tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True)


def get_english_test_prompts():
    return [
        {"name": "business_meeting", "system": "You are writing an email to your supervisor.",
         "user": "Write a brief email to your supervisor about rescheduling tomorrow's meeting. 2-3 sentences."},
        {"name": "apology", "system": "You are writing an apology email.",
         "user": "Write an apology email to a colleague for missing a deadline. 2-3 sentences, sincere."},
        {"name": "request", "system": "You are asking your professor for help.",
         "user": "Write an email to your professor requesting research guidance next week. 2-3 sentences."},
    ]


STANFORD_PROMPT_TEMPLATES = {
    "neutral": "Rewrite the following request, keeping the exact same meaning. Output only the rewritten text, 1-3 sentences:\n\n{text}",
    "formal": "Rewrite the following request to be MORE FORMAL and polite. Output only the rewritten text, 1-3 sentences:\n\n{text}",
    "casual": "Rewrite the following request to be MORE CASUAL and informal. Output only the rewritten text, 1-3 sentences:\n\n{text}",
}


CN_PROMPT_TEMPLATES = {
    "neutral": "请改写以下句子，保持原意不变。只输出改写后的句子，不要解释：\n\n{text}",
    "formal": "请将以下句子改写得更正式、更有礼貌。只输出改写结果：\n\n{text}",
    "casual": "请将以下句子改写得更口语化、更随意。只输出改写结果：\n\n{text}",
}


def get_chinese_test_prompts(max_samples=33, stratify_binary=None, prompt_mode="neutral", corpus_json=None):
    path = corpus_json or CN_CORPUS_JSON
    if not os.path.exists(path):
        print(f"❌ 中文语料不存在: {path}")
        return []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ 中文语料加载失败: {e}")
        return []
    items = data.get("items", [])
    if not items:
        return []
    counts = {1: 0, 0: 0, -1: 0}
    if stratify_binary:
        per_class = max(1, max_samples // len(stratify_binary))
        limits = {b: per_class for b in stratify_binary}
    else:
        limits = None
    template = CN_PROMPT_TEMPLATES.get(prompt_mode, CN_PROMPT_TEMPLATES["neutral"])
    system = "你是中文改写助手。保持原句意思，按指令调整语气。只输出改写结果，不要解释。"
    prompts = []
    for it in items:
        binary = it.get("orig_binary", 0)
        if limits is not None and (binary not in limits or counts.get(binary, 0) >= limits[binary]):
            continue
        text = (it.get("text") or "").strip()
        if len(text) < 2:
            continue
        prompts.append({
            "name": it.get("id", f"cn_{len(prompts)}"),
            "system": system,
            "user": template.format(text=text),
            "orig_text": text,
            "orig_binary": binary,
        })
        counts[binary] = counts.get(binary, 0) + 1
        if len(prompts) >= max_samples:
            break
    print(f"📂 中文语料: {len(prompts)} 条 | prompt_mode={prompt_mode} (polite:{counts.get(1,0)} neutral:{counts.get(0,0)} impolite:{counts.get(-1,0)})")
    return prompts


def _is_heavy_technical(text):
    lower = text.lower()
    tech = ["compile", "function(", "debug", "syntax error", "include", "namespace", "template<"]
    return sum(1 for t in tech if t in lower) >= 2


def _analyze_politeness_strategies(all_results, strengths):
    try:
        import spacy
        from convokit import PolitenessStrategies, Utterance, Speaker
    except ImportError as e:
        print(f"⚠ PolitenessStrategies 跳过: 缺少依赖 {e}")
        return None

    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("⚠ PolitenessStrategies 跳过: 请先运行 python -m spacy download en_core_web_sm")
        return None

    ps = PolitenessStrategies()


    items = []
    for s_idx, r in enumerate(all_results):
        for alpha in strengths:
            text = (r["strengths"].get(alpha) or "").strip()
            items.append((alpha, s_idx, text, f"a{alpha}_s{s_idx}"))


    by_alpha = {a: {} for a in strengths}
    counts = {a: 0 for a in strengths}
    strategy_names = set()

    for alpha, s_idx, text, utt_id in items:
        if not text:
            continue
        utt = Utterance(id=utt_id, text=text[:1024], speaker=Speaker(id="gen"))
        res = ps.transform_utterance(utt, spacy_nlp=nlp)
        feat = res.meta.get("politeness_strategies", {})
        strategy_names.update(feat.keys())
        for k, v in feat.items():
            if isinstance(v, (int, float)):
                by_alpha[alpha].setdefault(k, []).append(v)
        counts[alpha] += 1


    summary = {}
    for a in strengths:
        summary[a] = {}
        for k in strategy_names:
            vals = by_alpha[a].get(k, [])
            summary[a][k] = round(sum(vals) / len(vals), 4) if vals else 0.0


    POLITE = [
        "feature_politeness_==Please==", "feature_politeness_==Please_start==",
        "feature_politeness_==HASHEDGE==", "feature_politeness_==Hedges==",
        "feature_politeness_==Deference==", "feature_politeness_==Gratitude==",
        "feature_politeness_==Apologizing==", "feature_politeness_==HASPOSITIVE==",
        "feature_politeness_==SUBJUNCTIVE==", "feature_politeness_==Indirect_(btw)==",
    ]
    IMPOLITE = [
        "feature_politeness_==HASNEGATIVE==", "feature_politeness_==Direct_start==",
        "feature_politeness_==Direct_question==",
    ]
    composite = {}
    for a in strengths:
        s = summary[a]
        composite[a] = round(
            sum(s.get(k, 0) for k in POLITE) - sum(s.get(k, 0) for k in IMPOLITE), 4
        )

    return {
        "by_alpha": summary,
        "composite_by_alpha": composite,
        "strategy_names": sorted(strategy_names),
        "counts": counts,
    }


def get_stanford_corpus_prompts(max_samples=50, stratify_binary=None, prompt_mode="neutral", filter_technical=True):
    try:
        from convokit import Corpus, download
    except ImportError:
        print("❌ 请安装 convokit: pip install convokit")
        return []

    corpus = Corpus(filename=download("stack-exchange-politeness-corpus"))
    prompts = []
    counts = {1: 0, 0: 0, -1: 0}
    if stratify_binary:
        per_class = max(1, max_samples // len(stratify_binary))
        limits = {b: per_class for b in stratify_binary}
    else:
        limits = None

    template = STANFORD_PROMPT_TEMPLATES.get(prompt_mode, STANFORD_PROMPT_TEMPLATES["neutral"])
    system = "You are rewriting requests. Keep the meaning, optionally adjust the tone."

    for utt in corpus.iter_utterances():
        binary = utt.meta.get("Binary", 0)
        if limits is not None:
            if binary not in limits or counts.get(binary, 0) >= limits[binary]:
                continue
        text = (utt.text or "").strip()
        if len(text) < 20 or len(text) > 500:
            continue
        if filter_technical and _is_heavy_technical(text):
            continue

        prompts.append({
            "name": f"stanford_{utt.id}",
            "system": system,
            "user": template.format(text=text),
            "orig_text": text,
            "orig_binary": binary,
        })
        counts[binary] = counts.get(binary, 0) + 1
        if len(prompts) >= max_samples:
            break

    print(f"📂 Stanford Corpus: {len(prompts)} 条 | prompt_mode={prompt_mode} (polite:{counts.get(1,0)} neutral:{counts.get(0,0)} impolite:{counts.get(-1,0)})")
    return prompts


def run_cross_language_test(model_name="swallow", vector_index=1, max_english_tests=5,
                             strengths=None, corpus_source="builtin", stratify_binary=None,
                             prompt_mode="neutral", filter_technical=True,
                             intervention_layer=None, residual_type=None, fake_vector="none",
                             vector_source="japanese", allow_cross_lingual=False, vector_file_override=None,
                             vector_readout="truncated"):
    if strengths is None:
        strengths = [0.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0]
    layer = intervention_layer if intervention_layer is not None else INTERVENTION_LAYER
    res_type = residual_type or INTERVENTION_RESIDUAL_TYPE

    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    _MODEL_PATHS = {
        "swallow":    "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3",
        "elyza":      "elyza/Llama-3-ELYZA-JP-8B",
        "llama_en":   "meta-llama/Llama-3.1-8B-Instruct",
        "llama2":     "meta-llama/Llama-2-7b-chat-hf",
        "llama3_zh":  "shenzhi-wang/Llama3-8B-Chinese-Chat",
        "mistral":    "mistralai/Mistral-7B-Instruct-v0.3",
    }

    _BILINGUAL = {"swallow", "elyza", "llama3_zh"}

    _MONOLINGUAL_EN = {"llama_en", "llama2", "mistral"}

    model_path = _MODEL_PATHS.get(model_name, MODEL_PATH)
    if model_name in _MONOLINGUAL_EN:
        if vector_source != "english" and not allow_cross_lingual:
            vector_source = "english"

    if fake_vector != "none":
        vector_file = None
    elif vector_source == "english":
        import glob
        en_files = glob.glob(os.path.join(ENGLISH_VECTOR_DIR, f"english_formal_casual_{model_name}_layer*.pt"))
        vector_file = en_files[0] if en_files else None
        if vector_file is None:
            en_files = glob.glob(os.path.join(ENGLISH_VECTOR_DIR, "english_formal_casual_*.pt"))
            vector_file = en_files[0] if en_files else None
    elif vector_source == "chinese":
        import glob
        cn_files = glob.glob(os.path.join(CN_VECTOR_DIR, f"chinese_formal_casual_{model_name}_layer*.pt"))
        vector_file = cn_files[0] if cn_files else None
        if vector_file is None:
            cn_files = glob.glob(os.path.join(CN_VECTOR_DIR, "chinese_formal_casual_*.pt"))
            vector_file = cn_files[0] if cn_files else None
    else:
        if vector_file_override and os.path.exists(vector_file_override):
            vector_file = vector_file_override
        else:
            vector_file = os.path.join(FEATURE_VECTOR_DIR, f"swallow_argmax_detailed_{vector_index}_従業員_to_上司.pt")
            if not os.path.exists(vector_file):
                vector_file = os.path.join(FEATURE_VECTOR_DIR, FEATURE_VECTOR_FILENAME)

    output_lang = "cn" if corpus_source == "cn" else "en"
    print("=" * 60)
    if vector_source == "japanese":
        print("跨语言敬语向量迁移" + (" (中文输出)" if output_lang == "cn" else ""))
    elif vector_source == "chinese":
        print("中文 formal-casual 向量注入 (中文输出)")
    else:
        print("英语 formal-casual 向量注入")
    print("=" * 60)
    print(f"模型: {model_path}")
    print(f"向量: {vector_file}" if fake_vector == "none" else f"假向量: {fake_vector}")
    print(f"层: {layer} | 类型: {res_type}")
    print(f"强度: {strengths}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if vector_source == "english" and vector_file and fake_vector == "none":
        feature_vector = load_feature_vector_english(vector_file, layer, res_type)
        if feature_vector is not None:
            print(f"❷ 加载英语向量: {vector_file} (casual - formal)")
    elif vector_source == "chinese" and vector_file and fake_vector == "none":
        feature_vector = load_feature_vector_chinese(vector_file, layer, res_type)
        if feature_vector is not None:
            print(f"❷ 加载中文向量: {vector_file} (casual - formal)")
    else:
        feature_vector = load_feature_vector(vector_file, layer, res_type, fake_vector=fake_vector, readout=vector_readout)
    if feature_vector is None:
        return
    feature_vector = feature_vector.to(device)

    if corpus_source == "cn":
        prompts = get_chinese_test_prompts(
            max_samples=max_english_tests, stratify_binary=stratify_binary or [1, 0, -1],
            prompt_mode=prompt_mode,
        )
        if not prompts:
            return
    elif corpus_source == "stanford":
        prompts = get_stanford_corpus_prompts(
            max_samples=max_english_tests, stratify_binary=stratify_binary,
            prompt_mode=prompt_mode, filter_technical=filter_technical,
        )
        if not prompts:
            print("❌ Stanford corpus 加载失败，回退到内置 prompt")
            prompts = get_english_test_prompts()
    else:
        prompts = get_english_test_prompts()
    all_results = []

    lang_label = "中文" if output_lang == "cn" else "英语"
    for p_idx, prompt_data in enumerate(prompts[:max_english_tests]):
        print(f"\n{'='*60}")
        print(f"【{lang_label}测试 {p_idx+1}/{min(len(prompts), max_english_tests)}】 {prompt_data['name']}")
        print("=" * 60)

        messages = [
            {"role": "system", "content": prompt_data["system"]},
            {"role": "user", "content": prompt_data["user"]},
        ]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = tokenizer(prompt_text, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        input_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)

        sample_results = {"prompt": prompt_data, "strengths": {}}
        for alpha in strengths:
            out = generate_with_intervention(
                model, tokenizer, input_ids, attention_mask=attention_mask,
                intervention_vector=feature_vector if alpha != 0 else None,
                intervention_layer=layer if alpha != 0 else None,
                intervention_strength=alpha,
                residual_type=res_type,
            )
            label = "baseline" if alpha == 0 else f"α={alpha:+g}"
            print(f"  {label}: {out[:200]}{'...' if len(out) > 200 else ''}")
            sample_results["strengths"][alpha] = out
        all_results.append(sample_results)


    ps_result = None
    if output_lang == "en":
        ps_result = _analyze_politeness_strategies(all_results, strengths)
    if ps_result:
        print("\n" + "=" * 60)
        print("PolitenessStrategies 按 α 汇总")
        print("=" * 60)
        for a in strengths:
            comp = ps_result["composite_by_alpha"].get(a, 0)
            print(f"  α={a:+g}: composite={comp:+.4f}  (n={ps_result['counts'].get(a, 0)})")
        key_strategies = ["feature_politeness_==Please==", "feature_politeness_==Deference==",
                         "feature_politeness_==Apologizing==", "feature_politeness_==Gratitude==",
                         "feature_politeness_==Hedges=="]
        for ks in key_strategies:
            if ks in (ps_result["by_alpha"].get(strengths[0], {}) or {}):
                vals = [ps_result["by_alpha"].get(a, {}).get(ks, 0) for a in strengths]
                print(f"  {ks}: {[round(v, 3) for v in vals]}")


    formality_clf = _get_formality_clf(device) if output_lang == "en" else None

    formality_original = []
    formality_original_by_binary = {}  # {binary: [scores]}
    for r in all_results:
        orig_text = (r["prompt"].get("orig_text") or "").strip()
        orig_binary = r["prompt"].get("orig_binary")
        if not orig_text:
            continue
        if output_lang == "cn":
            p_orig = _cn_formality_score(orig_text)
        elif formality_clf:
            p_orig = _formality_prob(orig_text, formality_clf)
        else:
            p_orig = 0.5
        formality_original.append(p_orig)
        if orig_binary is not None:
            formality_original_by_binary.setdefault(orig_binary, []).append(p_orig)

    formality_by_alpha = {a: [] for a in strengths}
    formality_by_alpha_by_binary = {}  # {(alpha, binary): [scores]}
    for r in all_results:
        orig_binary = r["prompt"].get("orig_binary")
        for a in strengths:
            txt = (r["strengths"].get(a) or "").strip()
            if not txt:
                continue
            if output_lang == "cn":
                p = _cn_formality_score(txt)
            elif formality_clf:
                p = _formality_prob(txt, formality_clf)
            else:
                p = 0.5
            formality_by_alpha[a].append(p)
            if orig_binary is not None:
                formality_by_alpha_by_binary.setdefault((a, orig_binary), []).append(p)
    formality_result = None
    if formality_by_alpha and any(formality_by_alpha[a] for a in strengths):
        formality_result = {
            "mean_by_alpha": {a: round(sum(formality_by_alpha[a]) / len(formality_by_alpha[a]), 4)
                              if formality_by_alpha[a] else 0.5
                              for a in strengths},
            "counts": {a: len(formality_by_alpha[a]) for a in strengths},
            "evaluator": "cn_markers" if output_lang == "cn" else "roberta_formality_ranker",
        }

        if formality_original:
            formality_result["mean_original"] = round(sum(formality_original) / len(formality_original), 4)
            formality_result["count_original"] = len(formality_original)
        if formality_original_by_binary:
            formality_result["mean_by_binary_original"] = {
                f"binary={b}": round(sum(vals) / len(vals), 4)
                for b, vals in formality_original_by_binary.items() if vals
            }

        bins = sorted(set(b for (_, b) in formality_by_alpha_by_binary.keys()))
        if bins:
            formality_result["mean_by_alpha_by_binary"] = {}
            for a in strengths:
                for b in bins:
                    vals = formality_by_alpha_by_binary.get((a, b), [])
                    if vals:
                        formality_result["mean_by_alpha_by_binary"][f"α={a}_binary={b}"] = round(sum(vals) / len(vals), 4)
        print("\n" + "=" * 60)
        print("Formality 按 α 汇总" + (" (中文标记词)" if output_lang == "cn" else " (RoBERTa)"))
        print("=" * 60)
        if formality_result.get("mean_original") is not None:
            print(f"  原文 (baseline): P(formal)={formality_result['mean_original']:.4f}  (n={formality_result.get('count_original', 0)})")
        for a in strengths:
            m = formality_result["mean_by_alpha"].get(a, 0.5)
            print(f"  α={a:+g} (改写后): P(formal)={m:.4f}  (n={formality_result['counts'].get(a, 0)})")
        if formality_result.get("mean_by_binary_original"):
            print("  原文按 orig_binary 分层:", formality_result["mean_by_binary_original"])
        if formality_result.get("mean_by_alpha_by_binary"):
            print("\n  改写后按 orig_binary 分层:")
            for k, v in sorted(formality_result["mean_by_alpha_by_binary"].items()):
                print(f"    {k}: {v:.4f}")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if fake_vector != "none":
        vec_suffix = f"_fake_{fake_vector}"
    elif vector_source == "english":
        vec_suffix = "_en"
    elif vector_source == "chinese":
        vec_suffix = "_cn"
    else:
        vec_suffix = f"_v{vector_index}"
    out_pt = os.path.join(OUTPUT_DIR, f"cross_lang_{model_name}_{corpus_source}{vec_suffix}_{ts}.pt")
    out_json = os.path.join(OUTPUT_DIR, f"cross_lang_{model_name}_{corpus_source}{vec_suffix}_{ts}.json")


    metadata = {
        "transformers_version": __import__("transformers").__version__,
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "model_path": model_path,
        "random_seed": RANDOM_SEED,
        "vector_readout": vector_readout if vector_source == 'japanese' else 'stored',
        "vector_norm_fp32": float(feature_vector.float().norm()),
        "timestamp": datetime.datetime.now().isoformat(),
    }
    save_pt = {
        "metadata": metadata,
        "vector_index": vector_index,
        "vector_source": vector_file if fake_vector == "none" else f"fake_{fake_vector}",
        "model_name": model_name,
        "corpus_source": corpus_source,
        "prompt_mode": prompt_mode if corpus_source == "stanford" else None,
        "intervention_layer": layer,
        "residual_type": res_type,
        "fake_vector": fake_vector,
        "results": all_results,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    if ps_result:
        save_pt["politeness_strategies"] = ps_result
    if formality_result:
        save_pt["formality"] = formality_result
    torch.save(save_pt, out_pt)


    jsonable = []
    for r in all_results:
        entry = {
            "name": r["prompt"].get("name", ""),
            "orig_text": (r["prompt"].get("orig_text") or "")[:200],
            "orig_binary": r["prompt"].get("orig_binary"),
            "outputs": {str(k): (v[:400] + "..." if len(v) > 400 else v) for k, v in r["strengths"].items()},
        }
        jsonable.append(entry)
    json_out = {
        "metadata": metadata,
        "vector_index": vector_index,
        "strengths": strengths,
        "samples": jsonable,
        "intervention_layer": layer,
        "residual_type": res_type,
        "fake_vector": fake_vector,
    }
    if ps_result:
        json_out["politeness_strategies"] = {
            "composite_by_alpha": ps_result["composite_by_alpha"],
            "by_alpha": ps_result["by_alpha"],
            "counts": ps_result["counts"],
            "strategy_names": ps_result["strategy_names"],
        }
    if formality_result:
        json_out["formality"] = formality_result
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(json_out, f, ensure_ascii=False, indent=2)
    print(f"\n💾 结果已保存: {out_pt}")
    print(f"💾 可读 JSON: {out_json}")


def main():
    _fix_random_seed()
    parser = argparse.ArgumentParser(description='Evaluate transfer.')
    parser.add_argument("--model", type=str, default="swallow",
                        choices=["swallow", "elyza", "llama_en", "llama2", "llama3_zh", "mistral"],
                        help='Model alias or Hugging Face model path.')
    parser.add_argument("--vector_source", type=str, default="japanese", choices=["japanese", "english", "chinese"],
                        help='Extraction language.')
    parser.add_argument("--vector_index", type=int, default=1, help='Extraction sample index.')
    parser.add_argument("--vector_file", type=str, default=None,
                        help='Vector file path.')
    parser.add_argument('--vector-readout', choices=['truncated', 'full_mean'], default='truncated',
                        help='Original truncated native-precision mean or full-step FP32 geometry contrast.')
    parser.add_argument("--vector_indices", type=str, default=None,
                        help='Comma-separated extraction sample indices.')
    parser.add_argument("--max_tests", type=int, default=100, help='Maximum number of evaluation inputs.')
    parser.add_argument("--strengths", type=str, default="0,1,-1,2,-2,3,-3", help='Comma-separated intervention strengths.')
    parser.add_argument("--corpus", type=str, default="builtin", choices=["builtin", "stanford", "cn"],
                        help='Evaluation corpus.')
    parser.add_argument("--stratify", type=str, default=None,
                        help='Comma-separated class labels.')
    parser.add_argument("--prompt_mode", type=str, default="neutral", choices=["neutral", "formal", "casual"],
                        help='Evaluation instruction variant.')
    parser.add_argument("--no_filter_technical", action="store_true", help='No filter technical.')
    parser.add_argument("--allow_cross_lingual", action="store_true",
                        help='Allow cross lingual.')
    parser.add_argument("--layer", type=int, default=None, help='Zero-based layer index.')
    parser.add_argument("--residual_type", type=str, default=None, choices=["resid_pre", "resid_mid", "resid_post"],
                        help='Residual hook site.')
    parser.add_argument("--fake_vector", type=str, default="none", choices=["none", "zero", "random", "all"],
                        help='Optional control direction.')
    args = parser.parse_args()
    strengths = [float(x.strip()) for x in args.strengths.split(",")]
    stratify = None
    if args.stratify:
        stratify = [int(x.strip()) for x in args.stratify.split(",")]
    fake_modes = ["none", "zero", "random"] if args.fake_vector == "all" else [args.fake_vector]
    if args.vector_indices:
        indices = [int(x.strip()) for x in args.vector_indices.split(",")]
        for vi in indices:
            for fv in fake_modes:
                print(f"\n{'#'*60}\n向量索引 {vi} | fake_vector={fv}\n{'#'*60}\n")
                _fix_random_seed()
                run_cross_language_test(
                    args.model, vi, args.max_tests, strengths,
                    corpus_source=args.corpus, stratify_binary=stratify,
                    prompt_mode=args.prompt_mode, filter_technical=not args.no_filter_technical,
                    intervention_layer=args.layer, residual_type=args.residual_type, fake_vector=fv,
                    vector_source=args.vector_source, allow_cross_lingual=args.allow_cross_lingual,
                    vector_file_override=args.vector_file,
                    vector_readout=args.vector_readout,
                )
    else:
        for fv in fake_modes:
            print(f"\n{'#'*60}\nfake_vector={fv}\n{'#'*60}\n")
            _fix_random_seed()
            run_cross_language_test(
                args.model, args.vector_index, args.max_tests, strengths,
                corpus_source=args.corpus, stratify_binary=stratify,
                prompt_mode=args.prompt_mode, filter_technical=not args.no_filter_technical,
                intervention_layer=args.layer, residual_type=args.residual_type, fake_vector=fv,
                vector_source=args.vector_source, allow_cross_lingual=args.allow_cross_lingual,
                vector_file_override=args.vector_file,
                vector_readout=args.vector_readout,
            )


if __name__ == "__main__":
    main()
