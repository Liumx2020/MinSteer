#!/usr/bin/env python
# -*- coding: utf-8 -*-


import re
import torch
import os
import json
import random
import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM


_script_dir = os.path.dirname(os.path.abspath(__file__))
EMOTION_VECTOR_DIR = os.path.join(_script_dir, "extract_cached")
TWEET_DATA_DIR = os.path.join(_script_dir, "data")
TWEET_TEXT_FILE = os.path.join(TWEET_DATA_DIR, "test_text.txt")
TWEET_LABELS_FILE = os.path.join(TWEET_DATA_DIR, "test_labels.txt")
_script_name = os.path.splitext(os.path.basename(__file__))[0]
OUTPUT_DIR = os.path.join(_script_dir, _script_name)


SCENARIOS_3kvcache = ["office_worker", "student", "tourist", "propose", "chef"]

MODEL_PATH = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LAYER = 15
STRENGTH = 1.5
RESIDUAL_TYPE = "resid_post"
NUM_SAMPLES = 100
TWEET_SEED = 42
BOOTSTRAP_N = 1000
BOOTSTRAP_CI = 0.95
GENERATION_CONFIG = {"max_new_tokens": 150, "do_sample": False, "use_cache": True}
SENTIMENT_CLF_MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
MAPPING_FILE = os.path.join(TWEET_DATA_DIR, "mapping.txt")


def _load_label_mapping():
    label_name, label_to_id = {}, {}
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
            "text-classification", model=SENTIMENT_CLF_MODEL,
            device=dev, truncation=True, max_length=128,
        )
        _SENTIMENT_CLF_DEVICE = target
        print(f"[sentiment_clf] loaded {SENTIMENT_CLF_MODEL} on device={dev}")
    return _SENTIMENT_CLF

def _compute_content_preservation(cands, refs, lang="en", device="cuda:0"):
    try:
        from bert_score import score as bert_score_fn
    except ImportError:
        return None
    if not cands or not refs or len(cands) != len(refs):
        return None
    device_str = device if torch.cuda.is_available() else "cpu"
    _, _, F1 = bert_score_fn(cands, refs, lang=lang, verbose=False, device=device_str)
    return [round(float(f), 4) for f in F1.tolist()]

def _predict_sentiment(text, clf):
    if not text or not text.strip():
        return {"label": "neutral", "label_id": 1, "score": 0.5}
    new_text = []
    for t in text.split():
        t = "@user" if t.startswith("@") and len(t) > 1 else t
        t = "http" if t.startswith("http") else t
        new_text.append(t)
    preprocessed = " ".join(new_text)
    result = clf(preprocessed[:512])[0]
    raw_label, score = result["label"], result["score"]
    raw_lower = raw_label.lower().strip()
    label_str = RAW_LABEL_TO_NAME.get(raw_lower)
    if label_str is None and raw_lower in LABEL_TO_ID:
        label_str = raw_lower
    if label_str is None:
        m = re.search(r"label_?(\d)|^(\d)$", raw_lower)
        label_str = LABEL_NAME.get(int(m.group(1) or m.group(2)), "neutral") if m else "neutral"
    label_id = LABEL_TO_ID.get(label_str, 1)
    return {"label": label_str, "label_id": label_id, "score": round(score, 4)}

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
    std_val = (sum((x - mean_val) ** 2 for x in values) / (n - 1)) ** 0.5
    return {"mean": round(mean_val, 4), "std": round(std_val, 4), "ci_low": round(boot_means[lo], 4), "ci_high": round(boot_means[hi], 4)}


def find_emotion_vector_file_for_scenario(vector_dir, model_name, scenario, emotion_type, lang="en"):
    prefix = f"emotion_style_{model_name}_{lang}_{scenario}_{emotion_type}_"
    if not os.path.exists(vector_dir):
        return None
    for fn in os.listdir(vector_dir):
        if fn.startswith(prefix) and fn.endswith(".pt"):
            return os.path.join(vector_dir, fn)
    return None

@torch.no_grad()
def load_emotion_vector(file_path, layer_spec, residual_type):
    layer_indices = [layer_spec] if isinstance(layer_spec, int) else list(layer_spec)
    try:
        data = torch.load(file_path, map_location="cpu")
    except FileNotFoundError:
        return None, None, None
    residuals_g1 = data["generation_1"]["result"]["residuals"][residual_type]
    residuals_g2 = data["generation_2"]["result"]["residuals"][residual_type]
    layer_residuals_g1 = residuals_g1[:, layer_indices, :]
    layer_residuals_g2 = residuals_g2[:, layer_indices, :]
    avg_g1 = layer_residuals_g1.mean(dim=(0, 1))
    avg_g2 = layer_residuals_g2.mean(dim=(0, 1))
    vec = avg_g2 - avg_g1
    info = data.get("sample_info", {})
    return vec, info.get("initial_emotion"), info.get("target_emotion")

def load_emotion_vector_averaged(file_paths, layer_spec, residual_type):
    vectors = []
    for fp in file_paths:
        vec, _, _ = load_emotion_vector(fp, layer_spec, residual_type)
        if vec is not None:
            vectors.append(vec)
    if not vectors:
        return None, None, None
    avg_vec = torch.stack(vectors).mean(dim=0)
    vec0, init, tgt = load_emotion_vector(file_paths[0], layer_spec, residual_type)
    return avg_vec, init, tgt


def load_tweet_samples(num_samples, seed):
    with open(TWEET_TEXT_FILE, "r", encoding="utf-8") as f:
        texts = [line.strip().strip('"') for line in f if line.strip()]
    with open(TWEET_LABELS_FILE, "r", encoding="utf-8") as f:
        labels = [int(line.strip()) for line in f if line.strip()]
    if len(texts) != len(labels):
        raise ValueError("text/labels length mismatch")
    num_samples = min(num_samples, len(texts))
    if seed is not None:
        random.seed(seed)
    indices = random.sample(range(len(texts)), num_samples)
    return [(texts[i], labels[i]) for i in indices]

def create_rewrite_prompt(tweet_text, target_emotion):
    w = "positive" if target_emotion == "positive" else "negative"
    return f"Rewrite the following tweet in a {w} tone. Keep the same meaning.\n\nTweet: {tweet_text}"

def create_system_prompt_tweet_rewrite():
    return "You are an emotional style rewriting assistant. Rewrite the given tweet in the specified emotional tone while keeping the same meaning and information."

def generate_with_intervention(model, tokenizer, input_ids, intervention_vector=None,
        intervention_layer=15, intervention_strength=1.0, residual_type="resid_post"):
    device = next(model.parameters()).device
    vec = intervention_vector.to(device) if intervention_vector is not None else None
    hooks = []

    def hook_post(module, inputs, output):
        if vec is not None:
            delta = intervention_strength * vec
            out = output[0] if isinstance(output, tuple) else output
            out[:, -1, :] = out[:, -1, :] + delta
        return output

    if vec is not None and intervention_layer is not None:
        target = model.model.layers[intervention_layer]
        hook = target.register_forward_hook(hook_post)
        hooks.append(hook)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=GENERATION_CONFIG["max_new_tokens"],
            use_cache=GENERATION_CONFIG["use_cache"],
            do_sample=GENERATION_CONFIG["do_sample"],
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    for h in hooks:
        h.remove()
    return tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True)


def _run_single_vector_experiment(model, tokenizer, tweet_samples, emotion_vector,
        initial_emotion, target_emotion, sentiment_clf, device):
    system_prompt = create_system_prompt_tweet_rewrite()
    samples = []
    prompt_baseline_outputs = []
    prompt_baseline_preds = []

    for i, (content, orig_label) in enumerate(tweet_samples):
        if (i + 1) % 25 == 0 or i == 0:
            print(f"  Progress {i + 1}/{len(tweet_samples)}", flush=True)
        user_content = create_rewrite_prompt(content, target_emotion)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(prompt_text, return_tensors="pt").to(device)["input_ids"]

        baseline_out = generate_with_intervention(model, tokenizer, input_ids)
        prompt_baseline_outputs.append(baseline_out)
        pred_bl = _predict_sentiment(baseline_out, sentiment_clf)
        prompt_baseline_preds.append(pred_bl)

        out_neg = generate_with_intervention(model, tokenizer, input_ids,
            intervention_vector=emotion_vector, intervention_layer=LAYER,
            intervention_strength=STRENGTH, residual_type=RESIDUAL_TYPE)
        out_pos = generate_with_intervention(model, tokenizer, input_ids,
            intervention_vector=emotion_vector, intervention_layer=LAYER,
            intervention_strength=-STRENGTH, residual_type=RESIDUAL_TYPE)
        pred_neg = _predict_sentiment(out_neg, sentiment_clf)
        pred_pos = _predict_sentiment(out_pos, sentiment_clf)

        sk = str(STRENGTH)
        samples.append({
            "sample_idx": i + 1,
            "input_text": content,
            "orig_label": LABEL_NAME.get(orig_label, "?") if orig_label is not None else None,
            "prompt_baseline_output": baseline_out,
            "toward_negative_outputs": {sk: out_neg},
            "toward_positive_outputs": {sk: out_pos},
            "toward_negative_preds": {sk: {"label": pred_neg["label"], "score": pred_neg["score"]}},
            "toward_positive_preds": {sk: {"label": pred_pos["label"], "score": pred_pos["score"]}},
            "prompt_baseline_pred": {"label": pred_bl["label"], "score": pred_bl["score"]},
        })


    refs = [s["input_text"] for s in samples]
    dev_str = str(device)
    cp_baseline = _compute_content_preservation(prompt_baseline_outputs, refs, "en", dev_str)
    if cp_baseline:
        cp_neg = _compute_content_preservation([s["toward_negative_outputs"][str(STRENGTH)] for s in samples], refs, "en", dev_str)
        cp_pos = _compute_content_preservation([s["toward_positive_outputs"][str(STRENGTH)] for s in samples], refs, "en", dev_str)
        for i, s in enumerate(samples):
            s["content_preservation"] = {"prompt_baseline": cp_baseline[i], f"toward_negative_{STRENGTH}": cp_neg[i] if cp_neg else None, f"toward_positive_{STRENGTH}": cp_pos[i] if cp_pos else None}

    sk = str(STRENGTH)
    neg_hits = [1 if s["toward_negative_preds"][sk]["label"] == target_emotion else 0 for s in samples]
    pos_hits = [1 if s["toward_positive_preds"][sk]["label"] == initial_emotion else 0 for s in samples]
    bl_hits = [1 if (p and p.get("label") == target_emotion) else 0 for p in prompt_baseline_preds]
    classifier_summary = {
        "expected_direction": {"toward_negative": f"-> {target_emotion}", "toward_positive": f"-> {initial_emotion}"},
        "prompt_baseline": _bootstrap_ci(bl_hits, BOOTSTRAP_N, BOOTSTRAP_CI),
        "by_strength": {sk: {"toward_negative": _bootstrap_ci(neg_hits, BOOTSTRAP_N, BOOTSTRAP_CI), "toward_positive": _bootstrap_ci(pos_hits, BOOTSTRAP_N, BOOTSTRAP_CI)}},
    }
    content_preservation_summary = None
    if cp_baseline:
        content_preservation_summary = {
            "prompt_baseline": _bootstrap_ci(cp_baseline, BOOTSTRAP_N, BOOTSTRAP_CI),
            "by_strength": {sk: {"toward_negative": _bootstrap_ci(cp_neg, BOOTSTRAP_N, BOOTSTRAP_CI) if cp_neg else None, "toward_positive": _bootstrap_ci(cp_pos, BOOTSTRAP_N, BOOTSTRAP_CI) if cp_pos else None}},
        }
    return samples, classifier_summary, content_preservation_summary


def run_five_scenario_analysis(vector_dir=None):
    vector_dir = vector_dir or EMOTION_VECTOR_DIR
    model_name = "llama3"
    source_emotion = "positive"

    print("evaluate_scenarios: 5场景+平均, llama3 layer15 strength1.5, 100条/条件", flush=True)
    print("=" * 60, flush=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Loading model: {model_name} ({MODEL_PATH})", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16).to(device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    sentiment_clf = _get_sentiment_clf(device)

    tweet_samples = load_tweet_samples(NUM_SAMPLES, TWEET_SEED)
    print(f"Loaded {len(tweet_samples)} tweets (seed={TWEET_SEED})", flush=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


    scenario_files = []
    for sc in SCENARIOS_3kvcache:
        pf = find_emotion_vector_file_for_scenario(vector_dir, model_name, sc, source_emotion)
        if pf is None:
            print(f"WARNING: No vector file for scenario {sc}, skipping", flush=True)
        else:
            scenario_files.append((sc, pf))

    if not scenario_files:
        print("No scenario vector files found. Abort.")
        return

    results_all = {"runs": {}}


    for sc, filepath in scenario_files:
        print(f"\n{'#'*50}\nScenario: {sc}\n{'#'*50}", flush=True)
        vec, initial_emotion, target_emotion = load_emotion_vector(filepath, LAYER, RESIDUAL_TYPE)
        if vec is None:
            print(f"  Failed to load vector from {os.path.basename(filepath)}")
            continue
        print(f"  Vector: {initial_emotion} -> {target_emotion}, shape {tuple(vec.shape)}", flush=True)
        samples, clf_sum, cp_sum = _run_single_vector_experiment(
            model, tokenizer, tweet_samples, vec, initial_emotion, target_emotion, sentiment_clf, device
        )
        nd = clf_sum["by_strength"][str(STRENGTH)]["toward_negative"]
        pd = clf_sum["by_strength"][str(STRENGTH)]["toward_positive"]
        print(f"  toward_neg: {nd['mean']}±{nd['std']}, toward_pos: {pd['mean']}±{pd['std']}", flush=True)

        out_path = os.path.join(OUTPUT_DIR, f"scenario_{sc}_seed{TWEET_SEED}.json")
        result = {
            "meta": {"scenario": sc, "vector_file": os.path.abspath(filepath), "layer": LAYER, "strength": STRENGTH,
                     "num_samples": len(samples), "tweet_seed": TWEET_SEED, "initial_emotion": initial_emotion, "target_emotion": target_emotion},
            "samples": samples,
            "classifier_summary": clf_sum,
            "content_preservation_summary": cp_sum,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"  Saved: {out_path}", flush=True)
        results_all["runs"][sc] = {"file": out_path, "toward_neg_mean": nd["mean"], "toward_pos_mean": pd["mean"]}


    file_paths = [p for _, p in scenario_files]
    print(f"\n{'#'*50}\nScenario: averaged (5 vectors)\n{'#'*50}", flush=True)
    vec_avg, initial_emotion, target_emotion = load_emotion_vector_averaged(file_paths, LAYER, RESIDUAL_TYPE)
    if vec_avg is None:
        print("  Failed to load averaged vector")
    else:
        print(f"  Averaged vector: {initial_emotion} -> {target_emotion}, shape {tuple(vec_avg.shape)}", flush=True)
        samples, clf_sum, cp_sum = _run_single_vector_experiment(
            model, tokenizer, tweet_samples, vec_avg, initial_emotion, target_emotion, sentiment_clf, device
        )
        nd = clf_sum["by_strength"][str(STRENGTH)]["toward_negative"]
        pd = clf_sum["by_strength"][str(STRENGTH)]["toward_positive"]
        print(f"  toward_neg: {nd['mean']}±{nd['std']}, toward_pos: {pd['mean']}±{pd['std']}", flush=True)

        out_path = os.path.join(OUTPUT_DIR, f"scenario_averaged_seed{TWEET_SEED}.json")
        result = {
            "meta": {"scenario": "averaged", "vector_files": [os.path.abspath(p) for p in file_paths],
                     "layer": LAYER, "strength": STRENGTH, "num_samples": len(samples), "tweet_seed": TWEET_SEED,
                     "initial_emotion": initial_emotion, "target_emotion": target_emotion},
            "samples": samples,
            "classifier_summary": clf_sum,
            "content_preservation_summary": cp_sum,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"  Saved: {out_path}", flush=True)
        results_all["runs"]["averaged"] = {"file": out_path, "toward_neg_mean": nd["mean"], "toward_pos_mean": pd["mean"]}

    summary_path = os.path.join(OUTPUT_DIR, f"summary_seed{TWEET_SEED}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results_all, f, ensure_ascii=False, indent=2)
    print(f"\n{'='*50}\nSummary saved: {summary_path}", flush=True)

if __name__ == "__main__":
    run_five_scenario_analysis()
