"""Summarize the common-protocol sentiment extraction seeds."""
import argparse
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / 'results/sentiment_multiseed.json'
SEEDS = (42, 43, 44)
METHOD_N = {f'{method}_n{n}': n for method in ['minsteer', 'caa_class', 'repe']
            for n in [1, 5, 50] if (method, n) != ('repe', 1)}


def summarize(records):
    if set(records) != set(METHOD_N):
        raise ValueError('All eight method/budget rows are required')
    sizes = set()
    rows = {}
    for method, n in METHOD_N.items():
        group = sorted(records[method], key=lambda row: row['seed'])
        if [row['seed'] for row in group] != list(SEEDS):
            raise ValueError(f'Exactly one result per seed 42--44 is required: {method}')
        rates = []
        for row in group:
            total, positive = row['n_eval'], row['positive_count']
            if not isinstance(total, int) or not isinstance(positive, int) or not 0 <= positive <= total or not 1 <= total <= 500:
                raise ValueError(f'Invalid counts: {method}')
            sizes.add(total)
            rates.append(100 * positive / total)
        rows[method] = {'n': n, 'by_seed': group, 'mean_percent': statistics.mean(rates),
                        'sample_sd_percentage_points': statistics.stdev(rates)}
    if len(sizes) != 1:
        raise ValueError('All methods and seeds must use the same evaluation size')
    paired = {}
    for right in ['repe_n5', 'caa_class_n50']:
        differences = [100 * (a['positive_count'] - b['positive_count']) / a['n_eval']
                       for a, b in zip(rows['minsteer_n1']['by_seed'], rows[right]['by_seed'])]
        paired['minsteer_n1_minus_' + right] = {
            'by_seed': [{'seed': seed, 'difference_pp': difference} for seed, difference in zip(SEEDS, differences)],
            'mean_pp': statistics.mean(differences), 'sample_sd_pp': statistics.stdev(differences),
        }
    rows['repe_n1'] = {'n': 1, 'status': 'N/A', 'reason': 'One centered contrast has zero sample variance.'}
    return {'seeds': list(SEEDS), 'n_eval': sizes.pop(), 'layer': 15, 'alpha': 1.5,
            'evaluation': 'common_explicit_positive_tone',
            'caa_extraction': 'Positive/negative TweetEval training class texts; original INST template and second-last token; BF16 mean, no normalization.',
            'standard_deviation': 'Sample standard deviation across extraction seeds (ddof=1).',
            'rows': rows, 'paired_differences': paired}


def collect(directory):
    records = {}
    for method in METHOD_N:
        records[method] = []
        for seed in SEEDS:
            path = directory / f'{method}_seed{seed}_eval.json'
            data = json.loads(path.read_text())
            if (data['method'], data['seed'], data['n'], data['alpha']) != (method, seed, METHOD_N[method], 1.5):
                raise ValueError(f'Unexpected evaluation settings: {path}')
            if data['prompt_protocol'] != 'common_explicit_positive_tone':
                raise ValueError(f'Unexpected evaluation prompt: {path}')
            total = data['eval_size']
            positive = round(data['toward_pos'] * total)
            if not math.isclose(positive / total, data['toward_pos'], abs_tol=1e-12):
                raise ValueError(f'Rate does not correspond to an integer count: {path}')
            records[method].append({'seed': seed, 'n_eval': total, 'positive_count': positive})
    return summarize(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-reference', action='store_true')
    parser.add_argument('--input-dir', type=Path, default=ROOT / 'runs/sentiment_multiseed/evaluation')
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/sentiment_multiseed/summary.json')
    args = parser.parse_args()
    if args.verify_reference:
        reference = json.loads(REFERENCE.read_text())
        records = {method: reference['rows'][method]['by_seed'] for method in METHOD_N}
        assert summarize(records) == reference
        print('Verified eight method/budget rows across three extraction seeds; RepE n=1 is N/A.')
    else:
        result = collect(args.input_dir)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
