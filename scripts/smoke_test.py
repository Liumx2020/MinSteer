"""Run extraction, vector construction and evaluation on a small input prefix."""
import argparse
import csv
import gc
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_module(name, relative):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run(relative, *arguments):
    env = dict(os.environ)
    env['PYTHONPATH'] = str(ROOT) + os.pathsep + env.get('PYTHONPATH', '')
    subprocess.run([sys.executable, str(ROOT / relative), *map(str, arguments)],
                   cwd=ROOT, env=env, check=True)


def release_memory():
    import torch
    gc.collect()
    torch.cuda.empty_cache()


def sentiment(args):
    import torch
    run('scripts/prepare_data.py', 'sentiment')
    original = load_module('sentiment_extractor', 'sentiment/extract_cached.py')
    raw = args.output_dir / 'raw'
    raw.mkdir()
    original.DETAILED_RESIDUALS_OUTPUT_DIR = str(raw)
    calls = []
    generate = original.token_wise_generation_with_residuals

    def record(*positional, **keyword):
        calls.append(keyword.get('past_key_values') is not None)
        return generate(*positional, **keyword)

    original.token_wise_generation_with_residuals = record
    original.run_emotion_style_analysis(
        emotion_type='negative', sampling_method='argmax', top_p=0.9,
        temperature=0.7, max_samples=1, scenarios=['chef'], no_save=False, model='llama3')
    assert calls == [False, True]
    assert len(list(raw.glob('*.pt'))) == 1
    average = load_module('sentiment_average', 'experiments/sentiment_scaling/average.py')
    vector = average.average_vectors(1, str(raw))
    assert vector.shape == (32, 4096) and vector.dtype == torch.bfloat16
    vectors = args.output_dir / 'vectors' / 'n1'
    vectors.mkdir(parents=True)
    torch.save(vector[15], vectors / 'vec_layer_15_Meta-Llama-3.1-8B-Instruct.pt')
    release_memory()
    output = args.output_dir / 'evaluation.csv'
    run('experiments/sentiment_scaling/evaluate.py',
        '--n_values', 1, '--num_eval', args.num_eval, '--alpha_values', 1.5,
        '--vector_base_dir', vectors.parent, '--output_file', output)
    receipt = json.loads(output.with_suffix('.receipt.json').read_text())
    assert receipt['complete'] and receipt['row_count'] == args.num_eval
    with output.open() as stream:
        result = list(csv.DictReader(stream))
    return {'protocol': 'original plain-rewrite sentiment', 'extraction_pairs': 1,
            'stage_y_inherited_cache': calls[1], 'vector_dtype': str(vector.dtype),
            'evaluation_rows': args.num_eval, 'results': result}


def politeness(args):
    import torch
    run('scripts/prepare_data.py', 'politeness')
    original = load_module('politeness_extractor', 'politeness/extract_cached.py')
    raw = args.output_dir / 'raw'
    raw.mkdir()
    original.DETAILED_RESIDUALS_OUTPUT_DIR = str(raw)
    original.detailed_residual_analysis(model_name='swallow', max_prompts=1)
    files = sorted(raw.glob('*.pt'))
    assert len(files) == 1
    vector_path = args.output_dir / 'vector.pt'
    run('scripts/export_keigo_vector.py', files[0], vector_path)
    vector = torch.load(vector_path, map_location='cpu', weights_only=True)
    assert vector.shape == (4096,) and abs(vector.norm().item() - 1) < 1e-5
    release_memory()
    output = args.output_dir / 'evaluation.json'
    run('baselines/politeness.py',
        'evaluate', '--vector', vector_path, '--tag', 'minsteer_smoke',
        '--max-prompts', args.num_eval, '--strengths', -1, 0, 1, '--output', output)
    result = json.loads(output.read_text())
    assert len(result['rows']) == args.num_eval * 3
    assert all(row['p_formal'] is not None for row in result['rows'])
    return {'protocol': 'unit-norm copied-input politeness follow-up',
            'extraction_pairs': 1, 'extraction_scenario_index': 0,
            'evaluation_rows_per_strength': args.num_eval, 'strengths': [-1, 0, 1],
            'results': result['mean_p_formal']}


def toxicity(args):
    run('scripts/prepare_data.py', 'toxicity')
    sys.path.insert(0, str(ROOT / 'experiments/controls'))
    import toxicity as original
    original.OUT = args.output_dir / 'experiment'
    split = original.load_split()
    split['eval_samples'] = split['eval_samples'][:args.num_eval]
    original.load_split = lambda: split
    original.extract(SimpleNamespace(seed=42, n=1, max_new_tokens=80))
    release_memory()
    original.DOSES = (0.0, 0.4)
    original.evaluate(SimpleNamespace(seed=42, n=1, variant='minsteer',
                                     batch_size=16, window=10, max_new_tokens=60))
    path = original.OUT / 'eval/minsteer_seed42_n1_summary.json'
    result = json.loads(path.read_text())
    assert len(result) == 3 and all('mean_toxicity' in row for row in result)
    return {'protocol': 'matched lexical-token-mean toxicity follow-up',
            'extraction_pairs': 1, 'extraction_seed': 42,
            'evaluation_rows_per_strength': args.num_eval, 'strengths': [0, 0.4, -0.4],
            'results': result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', choices=['sentiment', 'politeness', 'toxicity'])
    parser.add_argument('--num-eval', type=int, default=10)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    maximum = {'sentiment': 500, 'politeness': 99, 'toxicity': 100}[args.task]
    if not 1 <= args.num_eval <= maximum:
        parser.error(f'--num-eval must be in [1, {maximum}]')
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        parser.error('Set CUDA_VISIBLE_DEVICES to an available GPU')
    import torch
    free, total = torch.cuda.mem_get_info()
    if total - free > 1024 ** 3:
        parser.error('The selected GPU is in use; choose an idle GPU')
    args.output_dir = (args.output_dir or ROOT / 'runs' / f'smoke_{args.task}').resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    result = globals()[args.task](args)
    result.update(task=args.task, elapsed_seconds=round(time.monotonic() - start, 2),
                  complete=True, paper_result_reproduction=False)
    (args.output_dir / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
