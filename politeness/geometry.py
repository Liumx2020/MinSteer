"""Keigo geometry with separate five-vector and single-reference statistics."""
import argparse
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def load_vector_all_layers(path, residual_type='resid_pre'):
    import torch
    data = torch.load(path, map_location='cpu', weights_only=False)
    if 'difference_vectors' in data:
        vector = data['difference_vectors'][residual_type].float()
        return vector.squeeze(1) if vector.ndim == 3 else vector
    a = data['generation_1']['result']['residuals'][residual_type].float()
    b = data['generation_2']['result']['residuals'][residual_type].float()
    return b.mean(0) - a.mean(0)


def cosine(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return float(a @ b / (a.norm() * b.norm() + 1e-9))


def summarize_vectors(jp, last, average, reference_index=1):
    if len(jp) != 5 or reference_index not in range(5):
        raise ValueError('Five JP vectors and a reference index from 0 to 4 are required')
    pairs = [cosine(jp[i], jp[j]) for i in range(5) for j in range(i + 1, 5)]
    last_cosines = [cosine(vector, last) for vector in jp]
    average_cosines = [cosine(vector, average) for vector in jp]
    return {
        'layer': 15, 'residual_type': 'resid_pre',
        'readout': 'FP32 full-step temporal means for JP; stored TwoSeq contrasts. No normalization.',
        'raw_l2_norms': {'minsteer_reference': float(jp[reference_index].float().norm()),
                         'twoseq_average': float(average.float().norm()), 'twoseq_last': float(last.float().norm())},
        'five_jp_comparison': {'jp_count': 5, 'last_cosines': last_cosines, 'average_cosines': average_cosines,
                               'last_mean': statistics.mean(last_cosines), 'average_mean': statistics.mean(average_cosines)},
        'single_jp_reference': {'index': reference_index, 'last_cosine': last_cosines[reference_index],
                                'average_cosine': average_cosines[reference_index]},
        'within_jp': {'pair_count': len(pairs), 'mean': statistics.mean(pairs),
                      'population_sd': statistics.pstdev(pairs), 'sample_sd': statistics.stdev(pairs)},
    }


def reported_statistics(summary):
    return {
        'raw_l2_norms': {name: round(value, 2) for name, value in summary['raw_l2_norms'].items()},
        'mean_cosine_to_five_jp': {'twoseq_last': round(summary['five_jp_comparison']['last_mean'], 2),
                                 'twoseq_average': round(summary['five_jp_comparison']['average_mean'], 2)},
        'single_jp_reference': {'index': summary['single_jp_reference']['index'],
                                'twoseq_last_cosine': round(summary['single_jp_reference']['last_cosine'], 4)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vector-dir', type=Path, default=ROOT / 'politeness/extract_cached')
    parser.add_argument('--reference-index', type=int, choices=range(5), default=1)
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/keigo_geometry/summary.json')
    parser.add_argument('--verify-reference', action='store_true')
    args = parser.parse_args()
    jp = [load_vector_all_layers(args.vector_dir / f'swallow_argmax_detailed_{i}_従業員_to_上司.pt')[15]
          for i in range(5)]
    last = load_vector_all_layers(args.vector_dir / 'swallow_argmax_last_token_0_従業員_to_上司.pt')[15]
    average = load_vector_all_layers(args.vector_dir / 'swallow_argmax_twoseq_avg_0_従業員_to_上司.pt')[15]
    result = summarize_vectors(jp, last, average, args.reference_index)
    result['reported'] = reported_statistics(result)
    if args.verify_reference:
        reference = json.loads((ROOT / 'results/keigo_geometry.json').read_text())
        assert result['reported'] == reference['reported'], 'Geometry differs from the manuscript reference'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['reported'], indent=2))


if __name__ == '__main__':
    main()
