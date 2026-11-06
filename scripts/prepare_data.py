"""Acquire upstream inputs and verify their historical ordering. No model inference."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(value):
    return hashlib.sha256(value).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=["sentiment", "toxicity", "politeness", "rtp"])
    args = parser.parse_args()
    contract = json.loads((ROOT / "configs/data_contract.json").read_text())
    if args.task == "sentiment":
        from datasets import load_dataset
        base = ROOT / "sentiment/data"
        base.mkdir(parents=True, exist_ok=True)
        for name, expected in contract["tweet_text_files"].items():
            url = "https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/sentiment/" + name
            content = urllib.request.urlopen(url).read()
            if sha(content) != expected:
                raise RuntimeError(f"Upstream {name} changed; do not normalize or substitute inputs")
            (base / name).write_bytes(content)
        ds = load_dataset("tweet_eval", "sentiment", split="test")
        got = [sha(json.dumps({"text": x["text"], "label": x["label"]}, ensure_ascii=False, sort_keys=True).encode()) for x in ds.select(range(500))]
        if got != contract["cached_eval500_row_sha256"]:
            raise RuntimeError("Cached HF dataset strings differ from the historical main evaluator")
        train = load_dataset("tweet_eval", "sentiment", split="train")
        pos = [x["text"] for x in train if x["label"] == 2]
        neg = [x["text"] for x in train if x["label"] == 0]
        pairs = [{"question": "Rewrite this tweet.", "answer_matching_behavior": f" {pos[i]}", "answer_not_matching_behavior": f" {neg[i]}"} for i in range(100)]
        content = json.dumps(pairs, indent=4).encode()
        if sha(content) != contract["caa_dataset_sha256"]:
            raise RuntimeError("CAA training pair order/text differs from the historical input")
        p = ROOT / "sentiment/relatedwork/CAA-main/datasets/generate/tweet_sentiment/generate_dataset.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)
    elif args.task in ("toxicity", "rtp"):
        sys.path.insert(0, str(ROOT / "toxicity"))
        import run_all_toxicity_eval as original
        original._lazy_imports()
        if args.task == "rtp":
            from datasets import load_dataset

            # Rebuild frozen membership without reclassifying prompts.
            ds = load_dataset("allenai/real-toxicity-prompts", split="train")
            wanted = {row["prompt"] for name in ("toxic_prompts", "nontoxic_prompts") for row in contract["rtp"][name]}
            found = {}
            for row in ds:
                prompt = row["prompt"]
                text = str(prompt["text"] if isinstance(prompt, dict) else prompt).strip()
                digest = sha(text.encode())
                if digest in wanted:
                    found[digest] = text
                if len(found) == len(wanted):
                    break
            if set(found) != wanted:
                raise RuntimeError("Upstream RTP data do not contain every frozen source prompt")
            pool = {"dataset": "allenai/real-toxicity-prompts", "seed": 42, "threshold": 0.5,
                    "n_toxic": 500, "n_nontoxic": 500,
                    "input_only": True, "selection": "historical frozen prompt membership"}
            for name in ("toxic_prompts", "nontoxic_prompts"):
                pool[name] = [{"prompt": found[row["prompt"]]} for row in contract["rtp"][name]]
            write_json(ROOT / "data/rtp_pool.json", pool)
            print("Prepared and verified RTP source prompts; no generated continuation is included.")
            return
        train, test = original.load_paradetox_splits()
        split = {"dataset": "s-nlp/paradetox", "seed": 42, "n_train_cands": 1000, "n_eval": 100,
                 "train_pairs_candidates": [{"toxic": a, "neutral": b} for a, b in train], "eval_samples": test}
        for name in ("train_pairs_candidates", "eval_samples"):
            got = [{k: sha(str(v).encode()) for k, v in row.items()} for row in split[name]]
            if got != contract["paradetox"][name]:
                raise RuntimeError(f"ParaDetox {name} differs from historical split")
        overlap = set(a["toxic"] for a in split["train_pairs_candidates"]) & set(a["toxic"] for a in test)
        if len(overlap) != contract["paradetox_duplicate_toxic_texts"]:
            raise RuntimeError("Unexpected change to historical repeated-text overlap")
        print(f"Historical index split retained: {len(overlap)} repeated toxic strings occur across its two blocks.")
        write_json(ROOT / "toxicity/run_all_toxicity_eval/toxicity_eval_results/paradetox_splits_seed42_train1000_eval100.json", split)
    else:
        from convokit import Corpus, download
        corpus = Corpus(filename=download("stack-exchange-politeness-corpus"))
        utterances = {str(x.id): x for x in corpus.iter_utterances()}
        samples = []
        for expected in contract["stanford99"]:
            uid = expected["name"].removeprefix("stanford_")
            # Match the saved 200-character source field.
            text = (utterances[uid].text or "").strip()[:200]
            if sha(text.encode()) != expected["text_sha256"]:
                raise RuntimeError(f"Stanford input {uid} differs from historical input")
            samples.append({"name": expected["name"], "orig_text": text, "orig_binary": expected["orig_binary"]})

        write_json(ROOT / "data/stanford99.json", {"samples": samples})
    print(f"Prepared and verified {args.task} inputs; local data remains ignored by git.")


if __name__ == "__main__":
    main()
