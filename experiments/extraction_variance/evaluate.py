"""Common positive-tone evaluation of the repeated-extraction vectors."""
import argparse
import json
from pathlib import Path

from summarize import METHOD_N, collect

ROOT = Path(__file__).resolve().parents[2]


def load_source_vector(directory, method, seed):
    import torch
    stem = f'{method}_seed{seed}'
    path = directory / (stem + '.pt')
    meta = json.loads((directory / (stem + '_extract.json')).read_text())
    if (meta['method'], meta['seed'], meta['n']) != (method, seed, METHOD_N[method]):
        raise ValueError(f'Unexpected vector metadata: {path}')
    vector = torch.load(path, map_location='cpu', weights_only=True)
    if meta['storage'] == 'unit':
        vector = vector.float() * meta['raw_norm']
    elif meta['storage'] != 'raw':
        raise ValueError(f'Unknown vector storage: {path}')
    return vector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['eval', 'aggregate'], default='eval')
    parser.add_argument('--methods', nargs='+', choices=list(METHOD_N), default=list(METHOD_N))
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    parser.add_argument('--layer', type=int, default=15)
    parser.add_argument('--alpha', type=float, default=1.5)
    parser.add_argument('--eval_tokens', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num-eval', type=int, choices=range(1, 501), default=500, metavar='1..500')
    parser.add_argument('--vector-dir', type=Path, default=ROOT / 'runs/sentiment_multiseed/vectors')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'runs/sentiment_multiseed/evaluation')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == 'aggregate':
        result = collect(args.output_dir)
        (args.output_dir / 'aggregate.json').write_text(json.dumps(result, indent=2) + '\n')
        return

    from extract import CLF_ID, eval_one, load_data, load_model_and_tokenizer
    from transformers import pipeline
    eval_set, _ = load_data()
    eval_set = eval_set[:args.num_eval]
    model, tokenizer = load_model_and_tokenizer()
    classifier = pipeline('sentiment-analysis', model=CLF_ID, device=0)
    for method in args.methods:
        for seed in args.seeds:
            output = args.output_dir / f'{method}_seed{seed}_eval.json'
            if output.exists():
                raise FileExistsError(output)
            vector = load_source_vector(args.vector_dir, method, seed)
            result = eval_one(method, seed, vector, model, tokenizer, classifier, eval_set, args)
            output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
