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


_DIR           = os.path.dirname(os.path.abspath(__file__))
TWEET_TEXT_FILE  = os.path.join(_DIR, "data", "test_text.txt")
TWEET_LABEL_FILE = os.path.join(_DIR, "data", "test_labels.txt")
OUTPUT_DIR       = os.path.join(_DIR, "repe_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)


MODEL_ID       = "meta-llama/Meta-Llama-3.1-8B-Instruct"
INJECT_LAYER   = 15
N_VALUES       = [1, 5, 10, 50]
ALPHAS         = [0.5, 1.5, 3.0]
EVAL_N         = 500
SEED           = 42
MAX_NEW_TOKENS = 150
SENTIMENT_CLF  = "cardiffnlp/twitter-roberta-base-sentiment"


def load_all_tweets():
    with open(TWEET_TEXT_FILE) as f:
        texts = [l.strip() for l in f]
    with open(TWEET_LABEL_FILE) as f:
        labels = [int(l.strip()) for l in f]
    return texts, labels


def split_eval_and_pool(texts, labels, eval_n=500, seed=42):
    """Shuffle, take first eval_n as evaluation set, rest as extraction pool."""
    rng = random.Random(seed)
    idx = list(range(len(texts)))
    rng.shuffle(idx)
    eval_idx = set(idx[:eval_n])
    eval_samples = [(texts[i], labels[i]) for i in idx[:eval_n]]
    pool_pos = [texts[i] for i in idx[eval_n:] if labels[i] == 2]
    pool_neg = [texts[i] for i in idx[eval_n:] if labels[i] == 0]
    return eval_samples, pool_pos, pool_neg


def build_pair_prompts(tokenizer, pos_tweet: str, neg_tweet: str):
    """Format a pos/neg tweet pair as chat prompts for RepE extraction."""
    def fmt(tweet, tone):
        msgs = [{"role": "user", "content":
                 f"This tweet expresses {tone} sentiment:\n{tweet}"}]
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False)
    return fmt(neg_tweet, "negative"), fmt(pos_tweet, "positive")


def extract_repe_direction(rep_pipeline, tokenizer, pool_pos, pool_neg,
                           n: int, layer: int, seed: int):
    """Sample n pairs from the pool and extract PCA direction."""
    rng = random.Random(seed + n)
    pos_sample = rng.sample(pool_pos, n)
    neg_sample = rng.sample(pool_neg, n)

    train_inputs = []
    train_labels = []
    for pos_tw, neg_tw in zip(pos_sample, neg_sample):
        neg_p, pos_p = build_pair_prompts(tokenizer, pos_tw, neg_tw)
        train_inputs.extend([neg_p, pos_p])   # flat: [neg, pos, neg, pos, ...]
        train_labels.append([0, 1])

    print(f"  [extract] calling get_directions with {len(train_inputs)} prompts ...", flush=True)
    rep_reader = rep_pipeline.get_directions(
        train_inputs,
        rep_token=-1,
        hidden_layers=[layer],
        n_difference=1,
        batch_size=8,
        train_labels=train_labels,
        direction_method="pca",
    )
    print(f"  [extract] done.", flush=True)

    direction = torch.tensor(rep_reader.directions[layer][0], dtype=torch.float32)
    sign      = rep_reader.direction_signs[layer][0]
    return direction * float(sign)


def load_sentiment_clf():
    from transformers import pipeline as hf_pipeline
    return hf_pipeline("text-classification", model=SENTIMENT_CLF,
                       device=0 if torch.cuda.is_available() else -1)


def classify_sentiment(clf, text: str) -> int:
    label_map = {"LABEL_0": 0, "LABEL_1": 1, "LABEL_2": 2}
    return label_map.get(clf(text[:512], truncation=True)[0]["label"], 1)


def generate_with_injection(model, tokenizer, input_ids, vector, layer, alpha):
    vec   = vector.to(model.device)
    hooks = []

    def hook_post(module, inputs, output):
        delta = alpha * vec
        if isinstance(output, tuple):
            output[0][:, -1, :] += delta
        else:
            output[:, -1, :] += delta
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
    total = len(samples)
    for i, (text, _) in enumerate(samples):
        if i % 50 == 0:
            print(f"    [{i}/{total}] hits={hits}", flush=True)
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
    parser.add_argument("--layer",  type=int, default=INJECT_LAYER)
    parser.add_argument("--n_eval", type=int, default=EVAL_N)
    parser.add_argument("--seed",   type=int, default=SEED)
    parser.add_argument("--alpha",  type=float, default=None,
                        help="Run only this alpha (default: all)")
    parser.add_argument("--n",      type=int, default=None,
                        help="Run only this n value (default: all N_VALUES)")
    args = parser.parse_args()

    n_values = [args.n] if args.n is not None else N_VALUES
    alphas   = [args.alpha] if args.alpha else ALPHAS
    print(f"[config] n_values={n_values}, layer={args.layer}, "
          f"n_eval={args.n_eval}, alphas={alphas}", flush=True)
    print(f"[repe]   repo: {_REPE_REPO}", flush=True)


    texts, labels = load_all_tweets()
    eval_samples, pool_pos, pool_neg = split_eval_and_pool(
        texts, labels, eval_n=args.n_eval, seed=args.seed)
    print(f"[data]   eval={len(eval_samples)}, "
          f"pool_pos={len(pool_pos)}, pool_neg={len(pool_neg)}", flush=True)
    assert len(pool_pos) >= max(N_VALUES), "Not enough positive samples"
    assert len(pool_neg) >= max(N_VALUES), "Not enough negative samples"


    print(f"\n[model] Loading {MODEL_ID} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model     = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, device_map="auto")
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    rep_pipeline = pipeline("rep-reading", model=model, tokenizer=tokenizer)
    clf = load_sentiment_clf()

    results = []
    for n in n_values:
        print(f"\n{'='*50}", flush=True)
        print(f"[RepE] n={n} pairs — extracting direction ...", flush=True)
        direction = extract_repe_direction(
            rep_pipeline, tokenizer, pool_pos, pool_neg,
            n=n, layer=args.layer, seed=args.seed)
        norm = direction.norm().item()
        print(f"  direction norm: {norm:.4f}", flush=True)

        vec_path = os.path.join(OUTPUT_DIR,
            f"repe_scaling_direction_n{n}_layer{args.layer}.pt")
        torch.save(direction, vec_path)
        print(f"  direction saved → {vec_path}", flush=True)

        for alpha in alphas:
            print(f"  [eval] alpha={alpha}, toward_positive ...", flush=True)
            rate_pos = evaluate(model, tokenizer, clf, eval_samples,
                                direction, args.layer, alpha, "positive")
            print(f"  [eval] alpha={alpha}, toward_negative ...", flush=True)
            rate_neg = evaluate(model, tokenizer, clf, eval_samples,
                                direction, args.layer, alpha, "negative")
            print(f"  [result] n={n} alpha={alpha}  toward_pos={rate_pos:.3f}  toward_neg={rate_neg:.3f}", flush=True)
            results.append({
                "method": "RepE (PCA, TweetEval pairs)",
                "n": n, "layer": args.layer, "alpha": alpha,
                "toward_pos": round(rate_pos, 4),
                "toward_neg": round(rate_neg, 4),
            })

    n_tag   = f"n{n_values[0]}" if len(n_values) == 1 else "all"
    out_csv = os.path.join(OUTPUT_DIR, f"repe_scaling_{n_tag}_layer{args.layer}.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[done] Results → {out_csv}")

    print("\n" + "="*60)
    print(f"RepE (PCA, TweetEval pairs) — toward_pos @ alpha=1.5")
    print("="*60)
    print(f"{'n':>5}  {'alpha':>6}  {'toward_pos':>12}  {'toward_neg':>12}")
    for r in results:
        print(f"{r['n']:>5}  {r['alpha']:>6.1f}  {r['toward_pos']:>12.1%}  {r['toward_neg']:>12.1%}")


if __name__ == "__main__":
    main()
