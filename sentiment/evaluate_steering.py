#!/usr/bin/env python
# -*- coding: utf-8 -*-


import re
import torch
import os
import json
import argparse
import random
import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM


_script_dir = os.path.dirname(os.path.abspath(__file__))
EMOTION_VECTOR_DIR = os.path.join(_script_dir, "extract_cached")


TWEET_DATA_DIR = os.path.join(_script_dir, "data")
TWEET_TEXT_FILE = os.path.join(TWEET_DATA_DIR, "test_text.txt")
TWEET_LABELS_FILE = os.path.join(TWEET_DATA_DIR, "test_labels.txt")
TWEET_TRAIN_TEXT_FILE = os.path.join(TWEET_DATA_DIR, "train_text.txt")
TWEET_TRAIN_LABELS_FILE = os.path.join(TWEET_DATA_DIR, "train_labels.txt")


_script_name = os.path.splitext(os.path.basename(__file__))[0]
OUTPUT_DIR = os.path.join(_script_dir, _script_name)


TEST_SCENARIOS = [
    "Please describe how you feel after finishing work today and briefly explain why.",
    "Share your feelings about the upcoming weekend.",
    "Tell me about your recent experience reading a book or watching a movie.",
    "Share your thoughts and feelings about the current weather.",
    "Describe your mood about starting a new semester or new project."
]


DEFAULT_INTERVENTION_LAYER = 15
DEFAULT_INTERVENTION_STRENGTH = 1.5
DEFAULT_RESIDUAL_TYPE = 'resid_post'
STRENGTH_ABLATION_VALUES = [0.5, 1.0, 1.5, 2.0]
BOOTSTRAP_N = 1000
BOOTSTRAP_CI = 0.95


GENERATION_CONFIG = {"max_new_tokens": 150, "do_sample": False, "use_cache": True}


MODEL_PATHS = {
    "llama3": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "qwen2.5": "Qwen/Qwen2.5-7B-Instruct",
    "gemma2": "google/gemma-2-9b-it",
}

MODEL_LAYER_CONFIG = {
    "llama3": {"values": [5, 10, 15, 20, 25, 30, "all"], "all_indices": [5, 10, 15, 20, 25, 30]},
    "qwen2.5": {"values": [4, 9, 13, 18, 22, 27, "all"], "all_indices": [4, 9, 13, 18, 22, 27]},
    "gemma2": {"values": [5, 12, 20, 28, 35, 41, "all"], "all_indices": [5, 12, 20, 28, 35, 41]},
}


SENTIMENT_CLF_MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
MAPPING_FILE = os.path.join(TWEET_DATA_DIR, "mapping.txt")


def _load_label_mapping():
    label_name = {}
    label_to_id = {}
    if os.path.exists(MAPPING_FILE):
        with open(MAPPING_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) >= 2:
                    lid = int(parts[0])
                    lname = parts[1].strip().lower()
                    label_name[lid] = lname
                    label_to_id[lname] = lid
    else:
        label_name = {0: "negative", 1: "neutral", 2: "positive"}
        label_to_id = {"negative": 0, "neutral": 1, "positive": 2}
    return label_name, label_to_id


LABEL_NAME, LABEL_TO_ID = _load_label_mapping()


RAW_LABEL_TO_NAME = {
    "label_0": "negative", "label_1": "neutral", "label_2": "positive",
    "label0": "negative", "label1": "neutral", "label2": "positive",
    "0": "negative", "1": "neutral", "2": "positive",
}


_SENTIMENT_CLF = None
_SENTIMENT_CLF_DEVICE = None


def _get_sentiment_clf(device):
    global _SENTIMENT_CLF, _SENTIMENT_CLF_DEVICE
    target = str(device)
    if _SENTIMENT_CLF is None or _SENTIMENT_CLF_DEVICE != target:
        from transformers import pipeline as hf_pipeline
        dev = 0 if torch.cuda.is_available() else -1
        _SENTIMENT_CLF = hf_pipeline(
            "text-classification",
            model=SENTIMENT_CLF_MODEL,
            device=dev,
            truncation=True,
            max_length=128,
        )
        _SENTIMENT_CLF_DEVICE = target
        print(f"[sentiment_clf] loaded {SENTIMENT_CLF_MODEL} on device={dev}")
    return _SENTIMENT_CLF


def _compute_content_preservation(cands: list, refs: list, lang="en", device="cuda:0"):
    try:
        from bert_score import score as bert_score_fn
    except ImportError:
        print("[content_preservation] 缺少 bert_score，请运行: pip install bert_score")
        return None
    if not cands or not refs or len(cands) != len(refs):
        return None
    device_str = device if torch.cuda.is_available() else "cpu"
    _, _, F1 = bert_score_fn(cands, refs, lang=lang, verbose=False, device=device_str)
    return [round(float(f), 4) for f in F1.tolist()]


def _predict_sentiment(text: str, clf) -> dict:
    if not text or not text.strip():
        return {"label": "neutral", "label_id": 1, "score": 0.5}

    new_text = []
    for t in text.split():
        t = "@user" if t.startswith("@") and len(t) > 1 else t
        t = "http" if t.startswith("http") else t
        new_text.append(t)
    preprocessed = " ".join(new_text)
    result = clf(preprocessed[:512])[0]
    raw_label = result["label"]
    score = result["score"]

    raw_lower = raw_label.lower().strip()
    label_str = RAW_LABEL_TO_NAME.get(raw_lower)
    if label_str is None and raw_lower in LABEL_TO_ID:
        label_str = raw_lower
    if label_str is None:
        m = re.search(r"label_?(\d)|^(\d)$", raw_lower)
        if m:
            idx = int(m.group(1) or m.group(2))
            label_str = LABEL_NAME.get(idx, "neutral")
        else:
            label_str = "neutral"
    label_id = LABEL_TO_ID.get(label_str, 1)
    return {"label": label_str, "label_id": label_id, "score": round(score, 4)}


def get_model_name_and_path(model=None):
    if model is not None:
        if model not in MODEL_PATHS:
            raise ValueError(f"Unsupported model: {model}. Use one of {list(MODEL_PATHS.keys())}")
        return model, MODEL_PATHS[model]
    return "llama3", MODEL_PATHS["llama3"]

def find_emotion_vector_files(vector_dir, model_name, emotion_type, lang="en"):
    matching_files = []

    if not os.path.exists(vector_dir):
        print(f"Error: Directory '{vector_dir}' does not exist")
        return matching_files


    prefix = f"emotion_style_{model_name}_{lang}_"

    for filename in os.listdir(vector_dir):
        if not filename.endswith('.pt'):
            continue
        if not filename.startswith(prefix):
            continue


        parts = filename[len(prefix):].split('_')


        if emotion_type in parts[:3]:
            matching_files.append(os.path.join(vector_dir, filename))

    return matching_files

@torch.no_grad()
def load_emotion_vector(file_path, layer_spec, residual_type):
    layer_indices = [layer_spec] if isinstance(layer_spec, int) else list(layer_spec)
    desc = f"layer {layer_indices[0]}" if len(layer_indices) == 1 else f"layers {layer_indices} (averaged)"
    print(f"Loading emotion function vector from file: {os.path.basename(file_path)} ({desc})")
    try:
        data = torch.load(file_path, map_location='cpu')
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        return None, None

    residuals_g1 = data['generation_1']['result']['residuals'][residual_type]
    residuals_g2 = data['generation_2']['result']['residuals'][residual_type]


    # residuals: [T, num_layers, H]
    layer_residuals_g1 = residuals_g1[:, layer_indices, :]  # [T, n, H]
    layer_residuals_g2 = residuals_g2[:, layer_indices, :]  # [T, n, H]
    avg_residual_g1 = layer_residuals_g1.mean(dim=(0, 1))  # [H]
    avg_residual_g2 = layer_residuals_g2.mean(dim=(0, 1))  # [H]

    emotion_vector = avg_residual_g2 - avg_residual_g1

    initial_emotion = data['sample_info']['initial_emotion']
    target_emotion = data['sample_info']['target_emotion']

    print(f"Emotion conversion vector calculated: {initial_emotion} -> {target_emotion}")
    print(f"Vector shape: {tuple(emotion_vector.shape)}")

    return emotion_vector, (initial_emotion, target_emotion)

def create_system_prompt():
    return "You are a human office worker sharing your daily experiences. Respond naturally as a real person would."

def create_system_prompt_tweet_rewrite():
    return "You are an emotional style rewriting assistant. Rewrite the given tweet in the specified emotional tone while keeping the same meaning and information."

def _bootstrap_ci(values, n_bootstrap=1000, ci=0.95):
    n = len(values)
    if n == 0:
        return {"mean": 0, "std": 0, "ci_low": 0, "ci_high": 0}
    if n < 2:
        v = float(values[0])
        return {"mean": round(v, 4), "std": 0, "ci_low": round(v, 4), "ci_high": round(v, 4)}
    boot_means = []
    for _ in range(n_bootstrap):
        idx = [random.randint(0, n - 1) for _ in range(n)]
        boot_means.append(sum(values[i] for i in idx) / n)
    boot_means.sort()
    lo = int((1 - ci) / 2 * n_bootstrap)
    hi = int((1 - (1 - ci) / 2) * n_bootstrap)
    mean_val = sum(values) / n
    std_val = (sum((x - mean_val) ** 2 for x in values) / (n - 1)) ** 0.5 if n > 1 else 0
    return {
        "mean": round(mean_val, 4),
        "std": round(std_val, 4),
        "ci_low": round(boot_means[lo], 4),
        "ci_high": round(boot_means[hi], 4),
    }


def load_tweet_samples(text_path, labels_path, num_samples=5, seed=None):
    if not os.path.exists(text_path) or not os.path.exists(labels_path):
        raise FileNotFoundError(f"Tweet data not found: {text_path} or {labels_path}")
    with open(text_path, "r", encoding="utf-8") as f:
        texts = [line.strip().strip('"') for line in f if line.strip()]
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = [int(line.strip()) for line in f if line.strip()]
    if len(texts) != len(labels):
        raise ValueError(f"train_text({len(texts)}) and train_labels({len(labels)}) length mismatch")
    if num_samples > len(texts):
        num_samples = len(texts)
    if seed is not None:
        random.seed(seed)
    indices = random.sample(range(len(texts)), num_samples)
    return [(texts[i], labels[i]) for i in indices]

def create_rewrite_prompt(tweet_text, target_emotion):
    w = "positive" if target_emotion == "positive" else "negative"
    return f"Rewrite the following tweet in a {w} tone. Keep the same meaning.\n\nTweet: {tweet_text}"

def generate_with_intervention(
    model, tokenizer, input_ids,
    intervention_vector=None,
    intervention_layer=None,
    intervention_strength=1.0,
    residual_type='resid_post',
    intervention_layers=None,
):
    device = next(model.parameters()).device
    hooks = []
    vec = intervention_vector.to(device) if intervention_vector is not None else None

    def intervention_hook_pre(module, inputs):
        if len(inputs) > 0 and vec is not None:
            inputs[0][:, -1, :] = inputs[0][:, -1, :] + intervention_strength * vec
        return inputs

    def intervention_hook_post(module, inputs, output):
        if vec is not None:
            delta = intervention_strength * vec
            if isinstance(output, tuple):
                output[0][:, -1, :] = output[0][:, -1, :] + delta
            else:
                output[:, -1, :] = output[:, -1, :] + delta
        return output


    layers_to_inject = []
    if intervention_layers == "all":
        layers_to_inject = list(range(len(model.model.layers)))
    elif isinstance(intervention_layers, (list, tuple)):
        layers_to_inject = [int(l) for l in intervention_layers]
    elif intervention_layer is not None and isinstance(intervention_layer, int):
        layers_to_inject = [intervention_layer]

    if vec is not None and layers_to_inject:
        for layer_idx in layers_to_inject:
            if layer_idx < 0 or layer_idx >= len(model.model.layers):
                continue
            target_layer = model.model.layers[layer_idx]
            if residual_type == 'resid_pre':
                hook = target_layer.input_layernorm.register_forward_pre_hook(intervention_hook_pre)
            elif residual_type == 'resid_mid':
                hook = target_layer.post_attention_layernorm.register_forward_pre_hook(intervention_hook_pre)
            else:
                hook = target_layer.register_forward_hook(intervention_hook_post)
            hooks.append(hook)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=GENERATION_CONFIG["max_new_tokens"],
            use_cache=GENERATION_CONFIG["use_cache"],
            do_sample=GENERATION_CONFIG["do_sample"],
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

    for h in hooks:
        h.remove()

    return tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True)

def _run_single_layer_experiment(
    model, tokenizer, tweet_samples, system_prompt, vector_file,
    layer_key, load_layer_spec, inject_layers, residual_type, strengths_to_run,
    initial_emotion, target_emotion, sentiment_clf, device, verbose,
    model_name="llama3",
):
    emotion_vector, _ = load_emotion_vector(vector_file, load_layer_spec, residual_type)
    if emotion_vector is None:
        return None

    samples = []
    prompt_baseline_outputs = []
    prompt_baseline_preds = []

    for sample_idx, (content, orig_label) in enumerate(tweet_samples):
        user_content = create_rewrite_prompt(content, target_emotion)
        if verbose:
            print(f"\n[Layer {layer_key}] Sample {sample_idx + 1}/{len(tweet_samples)}")
        elif (sample_idx + 1) % 50 == 0 or sample_idx == 0:
            print(f"[Layer {layer_key}] Progress {sample_idx + 1}/{len(tweet_samples)}", flush=True)


        if model_name == "gemma2":
            messages = [
                {"role": "user", "content": f"{system_prompt}\n\n{user_content}"}
            ]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(prompt_text, return_tensors="pt").to(device)["input_ids"]

        prompt_baseline_output = generate_with_intervention(model, tokenizer, input_ids)
        prompt_baseline_outputs.append(prompt_baseline_output)
        pred_baseline = _predict_sentiment(prompt_baseline_output, sentiment_clf) if sentiment_clf else None
        prompt_baseline_preds.append(pred_baseline)

        toward_neg_outputs = {}
        toward_pos_outputs = {}
        pred_toward_neg_by_strength = {}
        pred_toward_pos_by_strength = {}

        use_all_layers = inject_layers == "all"
        for s in strengths_to_run:
            out_neg = generate_with_intervention(
                model, tokenizer, input_ids,
                intervention_vector=emotion_vector,
                intervention_layer=None if use_all_layers else inject_layers[0],
                intervention_layers="all" if use_all_layers else None,
                intervention_strength=s,
                residual_type=residual_type
            )
            toward_neg_outputs[str(s)] = out_neg
            pn = _predict_sentiment(out_neg, sentiment_clf) if sentiment_clf else None
            if pn is not None:
                pred_toward_neg_by_strength[str(s)] = {"label": pn["label"], "score": pn["score"]}

            out_pos = generate_with_intervention(
                model, tokenizer, input_ids,
                intervention_vector=emotion_vector,
                intervention_layer=None if use_all_layers else inject_layers[0],
                intervention_layers="all" if use_all_layers else None,
                intervention_strength=-s,
                residual_type=residual_type
            )
            toward_pos_outputs[str(s)] = out_pos
            pp = _predict_sentiment(out_pos, sentiment_clf) if sentiment_clf else None
            if pp is not None:
                pred_toward_pos_by_strength[str(s)] = {"label": pp["label"], "score": pp["score"]}

        sample_record = {
            "sample_idx": sample_idx + 1,
            "input_text": content,
            "orig_label": LABEL_NAME.get(orig_label, "?") if orig_label is not None else None,
            "prompt_baseline_output": prompt_baseline_output,
            "toward_negative_outputs": toward_neg_outputs,
            "toward_positive_outputs": toward_pos_outputs,
            "toward_negative_preds": pred_toward_neg_by_strength,
            "toward_positive_preds": pred_toward_pos_by_strength,
        }
        if pred_baseline is not None:
            sample_record["prompt_baseline_pred"] = {"label": pred_baseline["label"], "score": pred_baseline["score"]}
        samples.append(sample_record)

    return samples, prompt_baseline_outputs, prompt_baseline_preds


def run_emotion_intervention_experiment(
    source_emotion="positive",
    residual_type=DEFAULT_RESIDUAL_TYPE,
    scenario_idx=0,
    vector_dir=None,
    vector_file=None,
    use_tweet_dataset=False,
    num_tweet_samples=200,
    tweet_seed=42,
    model=None,
    skip_layers=None,
    layers_override=None,
    strengths_override=None,
    no_save=False,
):
    print("Emotion style function vector intervention validation experiment starting...", flush=True)
    print("=" * 60, flush=True)

    if use_tweet_dataset:
        lang = "en"
        text_file = TWEET_TEXT_FILE
        labels_file = TWEET_LABELS_FILE
        print(f"Using TweetEval/posneg dataset (test split). Language set to 'en'.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_name, model_path = get_model_name_and_path(model=model)
    layer_config = MODEL_LAYER_CONFIG[model_name]
    if layers_override is None:
        layer_ablation_values = layer_config["values"]
    else:
        layer_ablation_values = []
        for x in layers_override:
            if str(x).lower() == "all":
                layer_ablation_values.append("all")
            else:
                layer_ablation_values.append(int(x))
    all_layer_indices = layer_config["all_indices"]

    print(f"Loading model: {model_name} ({model_path})", flush=True)
    print("  Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("  Loading model weights (may take 1-2 min)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    vdir = vector_dir or EMOTION_VECTOR_DIR
    if vector_file:
        if not os.path.isfile(vector_file):
            print(f"Error: --vector_file '{vector_file}' is not a file.")
            return
        print(f"Using specified vector file: {vector_file}", flush=True)
    else:
        vector_files = find_emotion_vector_files(vdir, model_name, source_emotion)
        if not vector_files:
            print(f"No matching emotion vector files found in '{vdir}': {model_name}_{lang}_{source_emotion}")
            return
        vector_file = vector_files[0]
    sentiment_clf = _get_sentiment_clf(device)
    print(f"Classifier enabled: {SENTIMENT_CLF_MODEL}")

    strengths_to_run = STRENGTH_ABLATION_VALUES if strengths_override is None else [float(x) for x in strengths_override]
    verbose = num_tweet_samples <= 20
    skip_set = set(str(s) for s in (skip_layers or []))

    if use_tweet_dataset:
        tweet_samples = load_tweet_samples(
            text_file, labels_file,
            num_samples=num_tweet_samples, seed=tweet_seed
        )
        system_prompt = create_system_prompt_tweet_rewrite()
        print(f"Loaded {len(tweet_samples)} tweets. Model: {model_name}. Layers: {layer_ablation_values}, Strengths: {strengths_to_run}")
    else:
        scenario_idx = min(scenario_idx, len(TEST_SCENARIOS) - 1)
        tweet_samples = [(TEST_SCENARIOS[scenario_idx], None)]
        system_prompt = create_system_prompt()

    data = torch.load(vector_file, map_location='cpu')
    initial_emotion = data['sample_info']['initial_emotion']
    target_emotion = data['sample_info']['target_emotion']
    print(f"Vector: {initial_emotion} -> {target_emotion}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = {
        "meta": {
            "timestamp": datetime.datetime.now().isoformat(),
            "model_name": model_name,
            "model_path": model_path,
            "source_emotion": source_emotion,
            "initial_emotion": initial_emotion,
            "target_emotion": target_emotion,
            "residual_type": residual_type,
            "use_tweet_dataset": use_tweet_dataset,
            "num_samples": len(tweet_samples) if use_tweet_dataset else 0,
            "tweet_seed": tweet_seed,
            "data_split": "test",
            "layers": [str(l) for l in layer_ablation_values],
            "strengths": strengths_to_run,
            "classifier": SENTIMENT_CLF_MODEL,
            "content_preservation": True,
            "vector_file": os.path.abspath(vector_file),
            "system_prompt": system_prompt,
            "generation": dict(GENERATION_CONFIG),
        },
        "by_layer": {},
    }

    if not use_tweet_dataset:
        for layer_val in layer_ablation_values:
            layer_key = str(layer_val)
            if layer_key in skip_set:
                print(f"[Skip] layer {layer_key}")
                continue
            load_layer_spec = all_layer_indices if layer_val == "all" else int(layer_val)
            inject_layers = "all" if layer_val == "all" else [int(layer_val)]
            ret = _run_single_layer_experiment(
                model, tokenizer, tweet_samples, system_prompt, vector_file,
                layer_key, load_layer_spec, inject_layers, residual_type, strengths_to_run,
                initial_emotion, target_emotion, sentiment_clf, device, verbose,
                model_name=model_name,
            )
            if ret:
                samples, _, _ = ret
                results["by_layer"][layer_key] = {"samples": samples}
        if not no_save:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(OUTPUT_DIR, f"tweet_eval_{model_name}_seed{tweet_seed}_{ts}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"Results saved to: {out_path}")
        else:
            print("(no_save, 未保存)")
        return

    preview_printed = False
    for layer_val in layer_ablation_values:
        layer_key = str(layer_val)
        if layer_key in skip_set:
            print(f"[Skip] layer {layer_key}")
            continue
        load_layer_spec = all_layer_indices if layer_val == "all" else int(layer_val)
        inject_layers = "all" if layer_val == "all" else [int(layer_val)]

        print(f"\n{'#'*60}\nLayer ablation: {layer_key}\n{'#'*60}", flush=True)
        ret = _run_single_layer_experiment(
            model, tokenizer, tweet_samples, system_prompt, vector_file,
            layer_key, load_layer_spec, inject_layers, residual_type, strengths_to_run,
            initial_emotion, target_emotion, sentiment_clf, device, verbose,
            model_name=model_name,
        )
        if ret is None:
            results["by_layer"][layer_key] = {"error": "Failed to load emotion vector"}
            continue

        samples, prompt_baseline_outputs, prompt_baseline_preds = ret

        if not preview_printed:
            print(f"\n{'='*70}\n完整注入结果预览 (前5条, Layer {layer_key})\n{'='*70}", flush=True)
            for i, s in enumerate(samples[:5]):
                pred_bl = s.get("prompt_baseline_pred", {})
                sk = "1.5" if "1.5" in (s.get("toward_negative_outputs") or {}) else str(strengths_to_run[0])
                pred_neg = s.get("toward_negative_preds", {}).get(sk, {})
                pred_pos = s.get("toward_positive_preds", {}).get(sk, {})
                out_neg = s.get("toward_negative_outputs", {}).get(sk, "")
                out_pos = s.get("toward_positive_outputs", {}).get(sk, "")
                print(f"\n【Sample {i+1}】orig_label: {s.get('orig_label', '?')}", flush=True)
                print(f"  输入: {s.get('input_text', '')[:120]}{'...' if len(s.get('input_text', '')) > 120 else ''}", flush=True)
                print(f"  [baseline] {str(s.get('prompt_baseline_output', ''))[:150]}{'...' if len(str(s.get('prompt_baseline_output', ''))) > 150 else ''} (pred: {pred_bl.get('label', '?')})", flush=True)
                print(f"  [toward_negative {sk}] {str(out_neg)[:150]}{'...' if len(str(out_neg)) > 150 else ''} (pred: {pred_neg.get('label', '?')})", flush=True)
                print(f"  [toward_positive {sk}] {str(out_pos)[:150]}{'...' if len(str(out_pos)) > 150 else ''} (pred: {pred_pos.get('label', '?')})", flush=True)
            print(f"{'='*70}\n", flush=True)
            preview_printed = True

        baseline_hits = [1 if (p and p.get("label") == target_emotion) else 0 for p in prompt_baseline_preds]
        baseline_stats = _bootstrap_ci(baseline_hits, BOOTSTRAP_N, BOOTSTRAP_CI)

        classifier_summary = {
            "expected_direction": {
                "toward_negative": f"-> {target_emotion}",
                "toward_positive": f"-> {initial_emotion}",
            },
            "prompt_baseline": baseline_stats,
            "by_strength": {},
        }
        for strength in strengths_to_run:
            sk = str(strength)
            neg_hits = [1 if s.get("toward_negative_preds", {}).get(sk, {}).get("label") == target_emotion else 0 for s in samples]
            pos_hits = [1 if s.get("toward_positive_preds", {}).get(sk, {}).get("label") == initial_emotion else 0 for s in samples]
            classifier_summary["by_strength"][sk] = {
                "toward_negative": _bootstrap_ci(neg_hits, BOOTSTRAP_N, BOOTSTRAP_CI),
                "toward_positive": _bootstrap_ci(pos_hits, BOOTSTRAP_N, BOOTSTRAP_CI),
            }

        classifier_summary["by_orig_label"] = {}
        for ol in ["negative", "neutral", "positive"]:
            sub = [s for s in samples if s.get("orig_label") == ol]
            if len(sub) == 0:
                continue
            bl_hits = [1 if (p and p.get("label") == target_emotion) else 0 for i, p in enumerate(prompt_baseline_preds) if samples[i].get("orig_label") == ol]
            classifier_summary["by_orig_label"][ol] = {"n": len(sub), "prompt_baseline": _bootstrap_ci(bl_hits, BOOTSTRAP_N, BOOTSTRAP_CI), "by_strength": {}}
            for strength in strengths_to_run:
                sk = str(strength)
                neg_hits_sub = [1 if s.get("toward_negative_preds", {}).get(sk, {}).get("label") == target_emotion else 0 for s in sub]
                pos_hits_sub = [1 if s.get("toward_positive_preds", {}).get(sk, {}).get("label") == initial_emotion else 0 for s in sub]
                classifier_summary["by_orig_label"][ol]["by_strength"][sk] = {
                    "toward_negative": _bootstrap_ci(neg_hits_sub, BOOTSTRAP_N, BOOTSTRAP_CI),
                    "toward_positive": _bootstrap_ci(pos_hits_sub, BOOTSTRAP_N, BOOTSTRAP_CI),
                }

        print(f"[Layer {layer_key}] Classifier: prompt_baseline->{target_emotion}: {baseline_stats['mean']}±{baseline_stats['std']}")
        for strength in strengths_to_run:
            sk = str(strength)
            nd = classifier_summary["by_strength"][sk]["toward_negative"]
            pd = classifier_summary["by_strength"][sk]["toward_positive"]
            print(f"  strength={strength}: toward_neg {nd['mean']}±{nd['std']}, toward_pos {pd['mean']}±{pd['std']}")

        refs = [s["input_text"] for s in samples]
        dev_str = str(device)
        cp_baseline = _compute_content_preservation(prompt_baseline_outputs, refs, "en", dev_str)
        content_preservation_summary = None
        if cp_baseline is not None:
            baseline_cp_stats = _bootstrap_ci(cp_baseline, BOOTSTRAP_N, BOOTSTRAP_CI)
            content_preservation_summary = {
                "prompt_baseline": baseline_cp_stats,
                "by_strength": {},
            }
            for i, s in enumerate(samples):
                s["content_preservation"] = {"prompt_baseline": cp_baseline[i]}
            for strength in strengths_to_run:
                sk = str(strength)
                cp_neg = _compute_content_preservation([s["toward_negative_outputs"][sk] for s in samples], refs, "en", dev_str)
                cp_pos = _compute_content_preservation([s["toward_positive_outputs"][sk] for s in samples], refs, "en", dev_str)
                content_preservation_summary["by_strength"][sk] = {
                    "toward_negative": _bootstrap_ci(cp_neg, BOOTSTRAP_N, BOOTSTRAP_CI) if cp_neg else None,
                    "toward_positive": _bootstrap_ci(cp_pos, BOOTSTRAP_N, BOOTSTRAP_CI) if cp_pos else None,
                }
                for i, s in enumerate(samples):
                    s["content_preservation"][f"toward_negative_{sk}"] = cp_neg[i] if cp_neg else None
                    s["content_preservation"][f"toward_positive_{sk}"] = cp_pos[i] if cp_pos else None
            print(f"[Layer {layer_key}] Content preservation: baseline {baseline_cp_stats['mean']}±{baseline_cp_stats['std']}")

        results["by_layer"][layer_key] = {
            "samples": samples,
            "classifier_summary": classifier_summary,
            "content_preservation_summary": content_preservation_summary,
        }

    if no_save:
        print(f"\n{'='*50}\nAll layers completed. (no_save, 未保存)", flush=True)
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_filename = f"tweet_eval_{model_name}_seed{tweet_seed}_{ts}.json"
        out_path = os.path.join(OUTPUT_DIR, out_filename)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n{'='*50}\nAll layers completed. Results saved to: {out_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Emotion style function vector intervention (TweetEval/posneg, layer+strength ablation)")
    parser.add_argument("--source_emotion", type=str, default="positive", choices=["positive", "negative"], help="Source emotion type")
    parser.add_argument("--residual_type", type=str, default=DEFAULT_RESIDUAL_TYPE, choices=["resid_pre", "resid_mid", "resid_post"], help="Residual type")
    parser.add_argument("--scenario", type=int, default=0, help="Test scenario index (used only when not --use_tweet_dataset)")
    parser.add_argument("--vector_dir", type=str, default=None, help="Override: path to emotion vector .pt files")
    parser.add_argument("--vector_file", type=str, default=None, help="Exact path to one .pt (for contrastive eval: KV vs twoseq_avg vs last_token)")
    parser.add_argument("--model", type=str, default=None, choices=list(MODEL_PATHS.keys()),
                        help='Model alias or Hugging Face model path.')
    parser.add_argument("--use_tweet_dataset", action="store_true", help="Use TweetEval/posneg (test split, 200 samples, seed 42)")
    parser.add_argument("--num_tweet_samples", type=int, default=200, help="Number of tweets when --use_tweet_dataset")
    parser.add_argument("--tweet_seed", type=int, default=42, help="Random seed for tweet sampling")
    parser.add_argument("--skip_layers", type=str, nargs="+", default=None,
                        help='Skip layers.')
    parser.add_argument("--layers", type=str, nargs="+", default=None,
                        help='Layers.')
    parser.add_argument("--strengths", type=float, nargs="+", default=None,
                        help='Comma-separated intervention strengths.')
    parser.add_argument("--no_save", action="store_true", help='Skip writing captured residuals.')

    args = parser.parse_args()

    run_emotion_intervention_experiment(
        source_emotion=args.source_emotion,
        residual_type=args.residual_type,
        scenario_idx=args.scenario,
        vector_dir=args.vector_dir,
        vector_file=args.vector_file,
        use_tweet_dataset=args.use_tweet_dataset,
        num_tweet_samples=args.num_tweet_samples,
        tweet_seed=args.tweet_seed,
        model=args.model,
        skip_layers=args.skip_layers,
        layers_override=args.layers,
        strengths_override=args.strengths,
        no_save=args.no_save,
    )

if __name__ == "__main__":
    main()
