"""Select, evaluate and summarize the five balanced n=5 sentiment subsets."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
POOL = ROOT / 'prompts/sentiment_pool50.json'
MANIFEST = ROOT / 'configs/sentiment_n5_subsets.json'
REFERENCE = ROOT / 'results/sentiment_n5_subsets.json'
SEEDS = (42, 43, 44, 45, 46)
VECTOR_NAME = 'vec_layer_15_Meta-Llama-3.1-8B-Instruct.pt'


def select_groups(pool):
    scenarios = sorted({row['scenario'] for row in pool})
    options = {s: sorted([r for r in pool if r['scenario'] == s],
                         key=lambda r: r['sample_idx']) for s in scenarios}
    assert len(scenarios) == 5 and all(len(rows) == 10 for rows in options.values())
    indices = {r['output_filename']: i for i, r in enumerate(pool)}
    groups = []
    for seed in SEEDS:
        rng = random.Random(seed)
        rows = [rng.choice(options[s]) for s in scenarios]
        choices = [{'scenario': r['scenario'], 'sample_idx': r['sample_idx'],
                    'pool_index': indices[r['output_filename']],
                    'output_filename': r['output_filename'],
                    'original_template': r['original_template']} for r in rows]
        groups.append({'selection_seed': seed, 'choices': choices})
    return groups


def make_manifest():
    raw = POOL.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    contract = json.loads((ROOT / 'configs/sentiment_scaling.json').read_text())
    assert digest == contract['prompt_pool_sha256']
    pool = json.loads(raw)
    return {'prompt_pool': 'prompts/sentiment_pool50.json', 'prompt_pool_sha256': digest,
            'selection_rule': 'Reset random.Random(seed); choose one row per lexicographically ordered scenario, with options sorted by sample_idx.',
            'selection_seeds': list(SEEDS), 'n': 5,
            'scenario_order': sorted({row['scenario'] for row in pool}),
            'groups': select_groups(pool)}


def checked_manifest():
    actual = json.loads(MANIFEST.read_text())
    if actual != make_manifest():
        raise ValueError('Frozen subset records differ from the selection rule or prompt pool')
    return actual


def summarize_groups(groups):
    groups = sorted(groups, key=lambda row: row['selection_seed'])
    if [row['selection_seed'] for row in groups] != list(SEEDS):
        raise ValueError('Exactly one result for each selection seed 42--46 is required')
    counts = {row['eval_count'] for row in groups}
    if len(counts) != 1 or not 1 <= next(iter(counts)) <= 500:
        raise ValueError('All groups must evaluate the same prefix of up to 500 inputs')
    clean = []
    for row in groups:
        count, positive = row['eval_count'], row['positive_count']
        if not isinstance(count, int) or not isinstance(positive, int) or not 0 <= positive <= count:
            raise ValueError('Evaluation and positive counts must be valid integers')
        rate = positive / count
        if 'positive_rate' in row and not math.isclose(rate, row['positive_rate'], abs_tol=1e-12):
            raise ValueError('Positive count and rate disagree')
        clean.append({'selection_seed': row['selection_seed'], 'eval_count': count,
                      'positive_count': positive, 'positive_rate': rate})
    rates = [row['positive_rate'] for row in clean]
    mean, sd = statistics.mean(rates), statistics.stdev(rates)
    return {'selection_manifest': 'configs/sentiment_n5_subsets.json',
            'metric': 'target_positive_rate', 'n': 5, 'alpha': 1.5, 'layer': 15,
            'evaluation': 'Same cached HF TweetEval test prefix for all five groups.',
            'groups': clean, 'mean_positive_rate': mean, 'sample_sd_positive_rate': sd,
            'standard_deviation': 'Sample standard deviation across five subsets (ddof=1), not standard error.',
            'reported_percent': {'mean': round(100 * mean, 1), 'sample_sd_pp': round(100 * sd, 1)}}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def build_vectors(raw_dir, output_dir, seeds):
    import torch
    torch.set_num_threads(2)
    groups = [g for g in checked_manifest()['groups'] if g['selection_seed'] in seeds]
    for group in groups:
        for choice in group['choices']:
            if not (raw_dir / choice['output_filename']).is_file():
                raise FileNotFoundError(raw_dir / choice['output_filename'])
    output_dir.mkdir(parents=True, exist_ok=False)
    cached = {}
    for group in groups:
        differences = []
        for choice in group['choices']:
            name = choice['output_filename']
            if name not in cached:
                data = torch.load(raw_dir / name, map_location='cpu', weights_only=False, mmap=True)
                a = data['generation_1']['result']['residuals']['resid_post']
                b = data['generation_2']['result']['residuals']['resid_post']
                assert a.dtype == b.dtype == torch.bfloat16
                assert a.ndim == b.ndim == 3 and a.shape[1:] == b.shape[1:]
                cached[name] = (b.mean(0) - a.mean(0)).clone()
                del data, a, b
            differences.append(cached[name])
        vector = torch.stack(differences).mean(0)
        assert vector.shape == (32, 4096) and vector.dtype == torch.bfloat16
        assert torch.isfinite(vector).all()
        directory = output_dir / f"seed{group['selection_seed']}"
        (directory / 'n5').mkdir(parents=True)
        torch.save(vector, directory / 'all_layers.pt')
        torch.save(vector[15], directory / 'n5' / VECTOR_NAME)
        print(f"Built seed {group['selection_seed']}: five BF16 contrasts, no normalization")


def evaluate(vector_dir, output_dir, seeds, num_eval):
    checked_manifest()
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('Set CUDA_VISIBLE_DEVICES to an idle GPU')
    for seed in seeds:
        if not (vector_dir / f'seed{seed}/n5' / VECTOR_NAME).is_file():
            raise FileNotFoundError(vector_dir / f'seed{seed}/n5' / VECTOR_NAME)
    output_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env['PYTHONPATH'] = str(ROOT) + os.pathsep + env.get('PYTHONPATH', '')
    for seed in seeds:
        output = output_dir / f'seed{seed}.csv'
        if output.exists() or output.with_suffix('.raw.jsonl').exists():
            raise FileExistsError(output)
        command = [sys.executable, str(Path(__file__).with_name('evaluate.py')),
                   '--n_values', '5', '--alpha_values', '1.5', '--layer', '15',
                   '--num_eval', str(num_eval), '--vector_base_dir', str(vector_dir / f'seed{seed}'),
                   '--output_file', str(output)]
        subprocess.run(command, cwd=ROOT, env=env, check=True)


def collect_results(input_dir):
    groups = []
    for seed in SEEDS:
        path = input_dir / f'seed{seed}.csv'
        with path.open() as stream:
            rows = list(csv.DictReader(stream))
        receipt = json.loads(path.with_suffix('.receipt.json').read_text())
        count = receipt['args']['num_eval']
        if not receipt.get('complete') or receipt['row_count'] != count or len(rows) != 1:
            raise ValueError(f'Incomplete or unexpected evaluation: {path}')
        settings = receipt['args']
        if settings['n_values'] != [5] or settings['alpha_values'] != [1.5] or settings['layer'] != 15:
            raise ValueError(f'Unexpected evaluation settings: {path}')
        row = rows[0]
        if int(row['n']) != 5 or float(row['alpha']) != 1.5:
            raise ValueError(f'Unexpected n or alpha: {path}')
        rate = float(row['pos_rate'])
        groups.append({'selection_seed': seed, 'eval_count': count,
                       'positive_count': round(rate * count), 'positive_rate': rate})
    return summarize_groups(groups)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('verify')
    select = sub.add_parser('select')
    select.add_argument('--output', type=Path, required=True)
    build = sub.add_parser('build')
    build.add_argument('--raw-dir', type=Path, default=ROOT / 'runs/sentiment_scaling/raw')
    build.add_argument('--output-dir', type=Path, default=ROOT / 'runs/sentiment_n5_subsets/vectors')
    build.add_argument('--selection-seeds', type=int, nargs='+', choices=SEEDS, default=list(SEEDS))
    run = sub.add_parser('evaluate')
    run.add_argument('--vector-dir', type=Path, default=ROOT / 'runs/sentiment_n5_subsets/vectors')
    run.add_argument('--output-dir', type=Path, default=ROOT / 'runs/sentiment_n5_subsets/evaluation')
    run.add_argument('--selection-seeds', type=int, nargs='+', choices=SEEDS, default=list(SEEDS))
    run.add_argument('--num-eval', type=int, choices=range(1, 501), default=500, metavar='1..500')
    summary = sub.add_parser('summarize')
    summary.add_argument('--input-dir', type=Path, default=ROOT / 'runs/sentiment_n5_subsets/evaluation')
    summary.add_argument('--output', type=Path, default=ROOT / 'runs/sentiment_n5_subsets/summary.json')
    args = parser.parse_args()
    if args.command == 'verify':
        checked_manifest()
        reference = json.loads(REFERENCE.read_text())
        assert summarize_groups(reference['groups']) == reference
        print(json.dumps(reference, indent=2))
    elif args.command == 'select':
        write_json(args.output, make_manifest())
    elif args.command == 'build':
        build_vectors(args.raw_dir.resolve(), args.output_dir.resolve(), args.selection_seeds)
    elif args.command == 'evaluate':
        evaluate(args.vector_dir.resolve(), args.output_dir.resolve(), args.selection_seeds, args.num_eval)
    else:
        result = collect_results(args.input_dir)
        write_json(args.output, result)
        print(json.dumps(result['reported_percent']))


if __name__ == '__main__':
    main()
