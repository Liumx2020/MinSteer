"""Repeated CAA extraction from positive and negative training texts."""
import argparse
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / 'runs/sentiment_multiseed/vectors'
ADAPTATION = ROOT / 'sentiment/relatedwork/CAA-main'


def class_pairs(train, n, seed):
    positive = [i for i, row in enumerate(train) if row['label'] == 2]
    negative = [i for i, row in enumerate(train) if row['label'] == 0]
    rng = random.Random(seed)
    pos_ids = rng.sample(positive, n)
    neg_ids = rng.sample(negative, n)
    pairs = [{'question': 'Rewrite this tweet.',
              'answer_matching_behavior': ' ' + train[p]['text'],
              'answer_not_matching_behavior': ' ' + train[q]['text']}
             for p, q in zip(pos_ids, neg_ids)]
    return pairs, {'positive_train_row_ids': pos_ids, 'negative_train_row_ids': neg_ids}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-values', type=int, nargs='+', choices=[1, 5, 50], default=[1, 5, 50])
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--layer', type=int, default=15)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('Set CUDA_VISIBLE_DEVICES to an idle GPU')
    import torch
    from datasets import load_dataset
    sys.path.insert(0, str(ADAPTATION))
    import generate_vectors as official
    from llama_wrapper import LlamaWrapper

    train = load_dataset('tweet_eval', 'sentiment', split='train')
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    wrapper = LlamaWrapper(os.environ.get('HF_TOKEN'), size='7b', use_chat=True)
    wrapper.model.eval()
    for n in args.n_values:
        for seed in args.seeds:
            stem = f'caa_class_n{n}_seed{seed}'
            destination = args.output_dir / (stem + '.pt')
            metadata = args.output_dir / (stem + '_extract.json')
            if destination.exists() or metadata.exists():
                raise FileExistsError(destination)
            directory = args.output_dir / stem
            directory.mkdir(exist_ok=False)
            pairs, selected = class_pairs(train, n, seed)
            data_path = directory / 'pairs.json'
            data_path.write_text(json.dumps(pairs, indent=4) + '\n')
            official.get_ab_data_path = lambda _behavior: str(data_path)
            official.get_vector_dir = lambda _behavior: str(directory)
            torch.manual_seed(seed)
            official.generate_save_vectors_for_behavior([args.layer], False, 'tweet_sentiment', wrapper, n)
            source = directory / f'n{n}/vec_layer_{args.layer}_Meta-Llama-3.1-8B-Instruct.pt'
            vector = torch.load(source, map_location='cpu', weights_only=True)
            assert vector.dtype == torch.bfloat16 and vector.shape == (4096,)
            torch.save(vector, destination)
            record = {'method': f'caa_class_n{n}', 'n': n, 'seed': seed, 'layer': args.layer,
                      'storage': 'raw', 'raw_norm': float(vector.float().norm()), **selected}
            metadata.write_text(json.dumps(record, indent=2) + '\n')
            print(f'Built {stem}: class texts, second-last token, raw BF16 mean', flush=True)


if __name__ == '__main__':
    main()
