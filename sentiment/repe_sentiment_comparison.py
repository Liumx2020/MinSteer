#!/usr/bin/env python3

import os, sys, argparse, csv, random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


_REPE_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "relatedwork", "representation-engineering")
if _REPE_REPO not in sys.path:
    sys.path.insert(0, _REPE_REPO)

from repe import repe_pipeline_registry
repe_pipeline_registry()


_DIR = os.path.dirname(os.path.abspath(__file__))
TWEET_DATA_DIR  = os.path.join(_DIR, "data")
TWEET_TEXT_FILE = os.path.join(TWEET_DATA_DIR, "test_text.txt")
TWEET_LABEL_FILE= os.path.join(TWEET_DATA_DIR, "test_labels.txt")
OUTPUT_DIR      = os.path.join(_DIR, "repe_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)


MODEL_ID       = "meta-llama/Meta-Llama-3.1-8B-Instruct"
INJECT_LAYER   = 15
ALPHAS         = [0.5, 1.5, 3.0]
EVAL_N         = 500
SEED           = 42
MAX_NEW_TOKENS = 150
SENTIMENT_CLF  = "cardiffnlp/twitter-roberta-base-sentiment"


SCENARIOS = [
    {
        "name": "office_worker",
        "system": "You are a human office worker sharing your daily experiences. Respond naturally.",
        "positive": "As an office worker, describe how your workday went today in a positive tone.",
        "negative": "As an office worker, describe how your workday went today in a negative tone.",
    },
    {
        "name": "student",
        "system": "You are a university student sharing your exam experiences. Respond naturally.",
        "positive": "As a student, describe how your exam went today in a positive tone.",
        "negative": "As a student, describe how your exam went today in a negative tone.",
    },
    {
        "name": "tourist",
        "system": "You are a tourist sharing your travel experiences. Respond naturally.",
        "positive": "As a tourist, describe your travel experience today in a positive tone.",
        "negative": "As a tourist, describe your travel experience today in a negative tone.",
    },
    {
        "name": "marriage_proposal",
        "system": "You are a person describing how a marriage proposal went. Respond naturally.",
        "positive": "Describe your marriage-proposal experience today in a positive tone.",
        "negative": "Describe your marriage-proposal experience today in a negative tone.",
    },
    {
        "name": "chef",
        "system": "You are a chef sharing your cooking experiences. Respond naturally.",
        "positive": "As a chef, describe your cooking experience today in a positive tone.",
        "negative": "As a chef, describe your cooking experience today in a negative tone.",
    },
]


def build_prompt(tokenizer, system: str, user: str) -> str:
    messages = [{"role": "system", "content": system},
                {"role": "user",   "content": user}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def extract_repe_direction_official(model, tokenizer, scenarios, layer: int, n_scenarios: int):
    print(f"\n[RepE-official] Extracting from {n_scenarios} scenario(s) at layer {layer} ...")

    train_inputs = []
    train_labels = []
    for sc in scenarios[:n_scenarios]:
        neg_p = build_prompt(tokenizer, sc["system"], sc["negative"])
        pos_p = build_prompt(tokenizer, sc["system"], sc["positive"])
        train_inputs.extend([neg_p, pos_p])
        train_labels.append([0, 1])
        print(f"  [{sc['name']}] prompts ready")


    rep_reading_pipeline = pipeline(
        "rep-reading",
        model=model,
        tokenizer=tokenizer,
    )

    rep_reader = rep_reading_pipeline.get_directions(
        train_inputs,
        rep_token=-1,               # last token
        hidden_layers=[layer],
        n_difference=1,
        batch_size=4,
        train_labels=train_labels,
        direction_method="pca",
    )

    # directions[layer] shape: (n_components, hidden_size); take PC1
    direction = torch.tensor(rep_reader.directions[layer][0], dtype=torch.float32)
    sign      = rep_reader.direction_signs[layer][0]
    direction = direction * float(sign)

    print(f"[RepE-official] Direction norm: {direction.norm().item():.4f}")
    return direction


def load_tweet_samples(n=500, seed=42):
    with open(TWEET_TEXT_FILE) as f:
        texts  = [l.strip() for l in f]
    with open(TWEET_LABEL_FILE) as f:
        labels = [int(l.strip()) for l in f]
    rng = random.Random(seed)
    idx = list(range(len(texts)))
    rng.shuffle(idx)
    return [(texts[i], labels[i]) for i in idx[:n]]


def load_sentiment_clf():
    from transformers import pipeline as hf_pipeline
    return hf_pipeline("text-classification", model=SENTIMENT_CLF,
                       device=0 if torch.cuda.is_available() else -1)


def classify_sentiment(clf, text: str) -> int:
    label_map = {"LABEL_0": 0, "LABEL_1": 1, "LABEL_2": 2}
    return label_map.get(clf(text[:512], truncation=True)[0]["label"], 1)


def generate_with_injection(model, tokenizer, input_ids, vector, layer: int, alpha: float):
    """Same resid_post last-token hook used in the MinSteer pipeline."""
    vec   = vector.to(model.device)
    hooks = []

    def hook_post(module, inputs, output):
        delta = alpha * vec
        if isinstance(output, tuple):
            output[0][:, -1, :] = output[0][:, -1, :] + delta
        else:
            output[:, -1, :]    = output[:, -1, :]    + delta
        return output

    hooks.append(model.model.layers[layer].register_forward_hook(hook_post))
    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=MAX_NEW_TOKENS,
                             do_sample=False, use_cache=True,
                             eos_token_id=tokenizer.eos_token_id,
                             pad_token_id=tokenizer.pad_token_id)
    for h in hooks:
        h.remove()
    return tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)


def evaluate(model, tokenizer, clf, samples, vector, layer, alpha, target_direction):
    sign         = +1.0 if target_direction == "positive" else -1.0
    target_label = 2    if target_direction == "positive" else 0
    hits = 0
    for text, _ in samples:
        msgs = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user",   "content":
                 f"Rewrite the following tweet in a {target_direction} tone. "
                 f"Keep the same meaning.\n\n{text}"}]
        prompt    = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        output    = generate_with_injection(model, tokenizer, input_ids, vector, layer, sign * alpha)
        if classify_sentiment(clf, output) == target_label:
            hits += 1
    return hits / len(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_scenarios", type=int, default=5)
    parser.add_argument("--layer",       type=int, default=INJECT_LAYER)
    parser.add_argument("--n_eval",      type=int, default=EVAL_N)
    parser.add_argument("--seed",        type=int, default=SEED)
    args = parser.parse_args()

    print(f"[config] n_scenarios={args.n_scenarios}, layer={args.layer}, "
          f"n_eval={args.n_eval}, alphas={ALPHAS}")
    print(f"[repe]   using official repo: {_REPE_REPO}")


    print(f"\n[model] Loading {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model     = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, device_map="auto")
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id


    direction = extract_repe_direction_official(
        model, tokenizer, SCENARIOS, args.layer, args.n_scenarios)

    vec_path = os.path.join(OUTPUT_DIR,
        f"repe_official_direction_n{args.n_scenarios}_layer{args.layer}.pt")
    torch.save(direction, vec_path)
    print(f"[RepE-official] Direction saved → {vec_path}")


    samples = load_tweet_samples(n=args.n_eval, seed=args.seed)
    print(f"\n[eval] {len(samples)} TweetEval samples (seed={args.seed})")
    clf = load_sentiment_clf()


    results = []
    for alpha in ALPHAS:
        print(f"\n[eval] alpha={alpha}")
        rate_pos = evaluate(model, tokenizer, clf, samples, direction, args.layer, alpha, "positive")
        rate_neg = evaluate(model, tokenizer, clf, samples, direction, args.layer, alpha, "negative")
        print(f"  toward_pos={rate_pos:.3f}  toward_neg={rate_neg:.3f}")
        results.append({"method": f"RepE-official (PCA, n={args.n_scenarios})",
                        "n_scenarios": args.n_scenarios, "layer": args.layer,
                        "alpha": alpha,
                        "toward_pos": round(rate_pos, 4),
                        "toward_neg": round(rate_neg, 4)})

    out_csv = os.path.join(OUTPUT_DIR,
        f"repe_official_n{args.n_scenarios}_layer{args.layer}.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[done] Results → {out_csv}")

    print("\n" + "="*60)
    print(f"RepE-official (PCA, n={args.n_scenarios}, layer={args.layer})")
    print("="*60)
    print(f"{'alpha':>6}  {'toward_pos':>12}  {'toward_neg':>12}")
    for r in results:
        print(f"{r['alpha']:>6.1f}  {r['toward_pos']:>12.1%}  {r['toward_neg']:>12.1%}")
    print()
    print("Reference (paper Table 2):")
    print("  MinSteer n=1,  alpha=1.5 → toward_pos=84.6%")


if __name__ == "__main__":
    main()
