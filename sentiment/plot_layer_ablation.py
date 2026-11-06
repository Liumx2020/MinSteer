"""Reproduce the sentiment layer plots from saved aggregate scores.

Data keys retain the original run aliases. The displayed name for ``llama3``
is Llama-3.1, matching the evaluated model. No scores are recomputed.
"""
from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


DISPLAY_NAMES = {'llama3': 'Llama-3.1'}
TOTAL_LAYERS = {'gemma2': 42, 'llama3': 32, 'qwen2.5': 28}


def plot_layer_ablation(layer_tables, output_dir, display_names=None):
    if display_names is None:
        display_names = DISPLAY_NAMES
    n_models = len(layer_tables)
    fig, axes = plt.subplots(
        2, n_models, figsize=(13, 5),
        gridspec_kw={'hspace': 0.35, 'wspace': 0.25},
    )
    if n_models == 1:
        axes = axes.reshape(2, -1)
    for idx, (model, rows) in enumerate(layer_tables.items()):
        layers = [str(r['layer']) for r in rows]
        neg = [r['toward_neg'] for r in rows]
        pos = [r['toward_pos'] for r in rows]
        x = np.arange(len(layers))
        axes[0, idx].bar(x, neg, color='steelblue', alpha=0.8, width=0.6)
        axes[0, idx].set_xticks(x)
        axes[0, idx].set_xticklabels(layers, rotation=45, ha='right', fontsize=8)
        axes[0, idx].set_ylabel('toward_neg', fontsize=9)
        axes[0, idx].set_title(display_names.get(model, model), fontsize=10, fontweight='bold')
        axes[0, idx].set_ylim(0, 1.05)
        axes[0, idx].grid(True, alpha=0.2, axis='y')
        axes[1, idx].bar(x, pos, color='coral', alpha=0.8, width=0.6)
        axes[1, idx].set_xticks(x)
        axes[1, idx].set_xticklabels(layers, rotation=45, ha='right', fontsize=8)
        axes[1, idx].set_ylabel('toward_pos', fontsize=9)
        axes[1, idx].set_title('(α=1.5)', fontsize=9)
        axes[1, idx].set_ylim(0, 1.05)
        axes[1, idx].grid(True, alpha=0.2, axis='y')
    fig.savefig(output_dir / 'layer_ablation.png', dpi=150, bbox_inches='tight')
    fig.savefig(output_dir / 'layer_ablation.pdf', bbox_inches='tight')
    plt.close(fig)


def relative_depth_data(layer_tables):
    return {
        model: [
            {'layer': int(row['layer']),
             'rel_depth': round(int(row['layer']) / TOTAL_LAYERS[model], 4),
             'toward_neg': row['toward_neg'], 'toward_pos': row['toward_pos']}
            for row in rows if row['layer'] != 'all'
        ]
        for model, rows in layer_tables.items()
    }


def plot_relative_depth(layer_tables, output_dir, display_names=None):
    if display_names is None:
        display_names = DISPLAY_NAMES
    cross_model_data = relative_depth_data(layer_tables)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for model, points in cross_model_data.items():
        x = [p['rel_depth'] for p in points]
        neg = [p['toward_neg'] for p in points]
        pos = [p['toward_pos'] for p in points]
        axes[0].plot(x, neg, 'o-', label=display_names.get(model, model))
        axes[1].plot(x, pos, 's-', label=display_names.get(model, model))
    axes[0].set_xlabel('Relative depth (layer / total_layers)')
    axes[0].set_ylabel('toward_negative (str=1.5)')
    axes[0].set_title('Cross-Model: toward_negative')
    axes[0].legend()
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel('Relative depth (layer / total_layers)')
    axes[1].set_ylabel('toward_positive (str=1.5)')
    axes[1].set_title('Cross-Model: toward_positive')
    axes[1].legend()
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / 'cross_model_layer.png', dpi=150, bbox_inches='tight')
    fig.savefig(output_dir / 'cross_model_layer.pdf', bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True,
                        help='JSON mapping original model aliases to layer scores')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(args.data.read_text())
    plot_layer_ablation(data, args.output_dir)
    plot_relative_depth(data, args.output_dir)


if __name__ == '__main__':
    main()
