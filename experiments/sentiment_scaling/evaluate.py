from recording import Recorder
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
import os
from transformers import pipeline


MODEL_PATH = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LAYER = 15
DEVICE = "cuda:0"
NUM_EVAL = 500
OUTPUT_FILE = "pure_caa_scaling_results.csv"


class SteeringHook:
    def __init__(self, vector, alpha, device):
        self.vector = vector.to(torch.bfloat16).to(device)
        self.alpha = alpha

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            return (output[0] + self.alpha * self.vector, ) + output[1:]
        return output + self.alpha * self.vector

import argparse

def run_scaling_test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_values", nargs="+", type=int, default=[1, 5, 10, 20, 50, 100])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_file", type=str, default="pure_caa_scaling_results.csv")
    parser.add_argument("--model_path", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument("--vector_base_dir", type=str, default="relatedwork/CAA-main/vectors/tweet_sentiment")
    parser.add_argument("--num_eval", type=int, default=500)
    parser.add_argument("--alpha_values", nargs="+", type=float, default=[0.5, 1.5, 3.0])
    args = parser.parse_args()
    audit = Recorder(args)

    device = args.device
    n_values = args.n_values
    output_file = args.output_file
    model_path = args.model_path
    layer = args.layer
    vector_base_dir = args.vector_base_dir
    num_eval = args.num_eval

    print(f"Loading model {model_path} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)

    audit.runtime(model, tokenizer)


    print("Loading sentiment classifier...")

    classifier = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment", device=0)
    label_map = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}


    ds = load_dataset("tweet_eval", "sentiment", split="test")
    eval_set = ds.select(range(num_eval))
    audit.dataset(ds, eval_set)


    alphas = args.alpha_values
    assert all(a in [0.5, 1.5, 3.0] for a in alphas)

    results = []

    for n in n_values:
        model_name_suffix = model_path.split("/")[-1]
        vector_path = f"{vector_base_dir}/n{n}/vec_layer_{layer}_{model_name_suffix}.pt"
        if not os.path.exists(vector_path):
            print(f"Vector for n={n} not found at {vector_path}, skipping...")
            continue

        print(f"\n===== Testing n={n} =====")
        steering_vector = torch.load(vector_path)
        audit.vector(vector_path, steering_vector)

        for alpha in alphas:
            print(f"\n--- n={n}, alpha={alpha} ---")
            hook_handle = model.model.layers[layer].register_forward_hook(SteeringHook(steering_vector, alpha, device))
            pos_count = 0
            results_list = []
            for i, item in enumerate(tqdm(eval_set)):
                prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nRewrite the following tweet: {item['text']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                inputs = tokenizer(prompt, return_tensors="pt").to(device)

                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=80,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )

                response = tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                pred = classifier(response)[0]
                audit.row(n, alpha, i, item, prompt, inputs, output_ids, response, pred)
                is_pos = label_map[pred['label']] == "positive"
                results_list.append(is_pos)
                if is_pos:
                    pos_count += 1

            hook_handle.remove()


            label_results = {}
            for l in [0, 1, 2]:
                l_indices = [i for i, item in enumerate(eval_set) if item['label'] == l]
                if not l_indices: continue
                l_pos = sum(1 for idx in l_indices if results_list[idx])
                label_results[f"pos_rate_orig_{l}"] = l_pos / len(l_indices)

            res_entry = {
                "n": n,
                "alpha": alpha,
                "pos_rate": pos_count / num_eval
            }
            res_entry.update(label_results)
            results.append(res_entry)


            pd.DataFrame(results).to_csv(output_file, index=False)

    print(f"\nFinal Scaling Results saved to {output_file}")
    print(pd.DataFrame(results))
    audit.finish(results)

if __name__ == "__main__":
    run_scaling_test()
