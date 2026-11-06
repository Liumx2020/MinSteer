
import json
import torch as t
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import os
from dotenv import load_dotenv
from llama_wrapper import LlamaWrapper
import argparse
from typing import List
from utils.tokenize import tokenize_llama_base, tokenize_llama_chat
from behaviors import (
    get_vector_dir,
    get_activations_dir,
    get_ab_data_path,
    get_vector_path,
    get_activations_path,
    ALL_BEHAVIORS
)

load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")


class ComparisonDataset(Dataset):
    def __init__(self, data_path, token, model_name_path, use_chat, **kwargs):
        with open(data_path, "r") as f:
            self.data = json.load(f)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_path, token=token
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.use_chat = use_chat


        num_samples = kwargs.get("num_samples", None)
        if num_samples is not None:
            self.data = self.data[:num_samples]

    def prompt_to_tokens(self, instruction, model_output):
        if self.use_chat:
            tokens = tokenize_llama_chat(
                self.tokenizer,
                user_input=instruction,
                model_output=model_output,
            )
        else:
            tokens = tokenize_llama_base(
                self.tokenizer,
                user_input=instruction,
                model_output=model_output,
            )
        return t.tensor(tokens).unsqueeze(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        p_text = item["answer_matching_behavior"]
        n_text = item["answer_not_matching_behavior"]
        q_text = item["question"]
        p_tokens = self.prompt_to_tokens(q_text, p_text)
        n_tokens = self.prompt_to_tokens(q_text, n_text)
        return p_tokens, n_tokens

def generate_save_vectors_for_behavior(
    layers: List[int],
    save_activations: bool,
    behavior: str,
    model: LlamaWrapper,
    num_samples: int = None,
):
    behavior_dir = get_vector_dir(behavior)
    if num_samples is not None:
        behavior_dir = os.path.join(behavior_dir, f"n{num_samples}")

    if not os.path.exists(behavior_dir):
        os.makedirs(behavior_dir)

    if save_activations:
        activations_dir = get_activations_dir(behavior)
        if num_samples is not None:
            activations_dir = os.path.join(activations_dir, f"n{num_samples}")
        if not os.path.exists(activations_dir):
            os.makedirs(activations_dir)

    model.set_save_internal_decodings(False)
    model.reset_all()

    pos_activations = dict([(layer, []) for layer in layers])
    neg_activations = dict([(layer, []) for layer in layers])

    dataset = ComparisonDataset(
        get_ab_data_path(behavior),
        HUGGINGFACE_TOKEN,
        model.model_name_path,
        model.use_chat,
        num_samples=num_samples,
    )

    for p_tokens, n_tokens in tqdm(dataset, desc="Processing prompts"):
        p_tokens = p_tokens.to(model.device)
        n_tokens = n_tokens.to(model.device)
        model.reset_all()
        model.get_logits(p_tokens)
        for layer in layers:
            p_activations = model.get_last_activations(layer)
            if len(p_activations.shape) == 2:
                p_activations = p_activations[-2, :].detach().cpu()
            else:
                p_activations = p_activations[0, -2, :].detach().cpu()
            pos_activations[layer].append(p_activations)
        model.reset_all()
        model.get_logits(n_tokens)
        for layer in layers:
            n_activations = model.get_last_activations(layer)
            if len(n_activations.shape) == 2:
                n_activations = n_activations[-2, :].detach().cpu()
            else:
                n_activations = n_activations[0, -2, :].detach().cpu()
            neg_activations[layer].append(n_activations)

    for layer in layers:
        all_pos_layer = t.stack(pos_activations[layer])
        all_neg_layer = t.stack(neg_activations[layer])
        vec = (all_pos_layer - all_neg_layer).mean(dim=0)


        vec_path = os.path.join(behavior_dir, f"vec_layer_{layer}_{model.model_name_path.split('/')[-1]}.pt")
        t.save(vec, vec_path)
        if save_activations:
            t.save(
                all_pos_layer,
                get_activations_path(behavior, layer, model.model_name_path, "pos"),
            )
            t.save(
                all_neg_layer,
                get_activations_path(behavior, layer, model.model_name_path, "neg"),
            )

def generate_save_vectors(
    layers: List[int],
    save_activations: bool,
    use_base_model: bool,
    model_size: str,
    behaviors: List[str],
    num_samples: int = None,
):
    model = LlamaWrapper(
        HUGGINGFACE_TOKEN, size=model_size, use_chat=not use_base_model
    )
    for behavior in behaviors:
        generate_save_vectors_for_behavior(
            layers, save_activations, behavior, model, num_samples
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", nargs="+", type=int, default=list(range(32)))
    parser.add_argument("--save_activations", action="store_true", default=False)
    parser.add_argument("--use_base_model", action="store_true", default=False)
    parser.add_argument("--model_size", type=str, choices=["7b", "13b", "qwen2.5", "gemma2"], default="7b")
    parser.add_argument("--behaviors", nargs="+", type=str, default=ALL_BEHAVIORS)
    parser.add_argument("--num_samples", type=int, default=None)

    args = parser.parse_args()
    generate_save_vectors(
        args.layers,
        args.save_activations,
        args.use_base_model,
        args.model_size,
        args.behaviors,
        args.num_samples
    )
