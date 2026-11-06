from release_paths import project_path
import os
import torch
import numpy as np
import glob
from collections import defaultdict

def get_style_vector(file_path):
    try:
        data = torch.load(file_path, map_location='cpu')

        res_types = data['generation_1']['result']['residuals'].keys()
        target_type = 'resid_post' if 'resid_post' in res_types else list(res_types)[0]

        res1 = data['generation_1']['result']['residuals'][target_type]
        res2 = data['generation_2']['result']['residuals'][target_type]

        # res shape: [seq, layers, hidden] or [1, layers, hidden]
        # In some files it might be [1, 1, layers, hidden] if it was unsqueezed twice
        while len(res1.shape) > 3:
            res1 = res1.squeeze(0)
            res2 = res2.squeeze(0)

        v1 = res1[-1] # [layers, hidden]
        v2 = res2[-1] # [layers, hidden]


        fname = os.path.basename(file_path)
        if 'positive' in fname:

            return v1 - v2
        else:

            return v2 - v1
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def compute_avg_cosine_similarity(vectors):
    # vectors: list of [layers, hidden]
    if len(vectors) < 2:
        return None

    num_layers = vectors[0].shape[0]
    all_sims = []

    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            v1 = vectors[i].float()
            v2 = vectors[j].float()

            norm1 = torch.norm(v1, dim=-1) # [layers]
            norm2 = torch.norm(v2, dim=-1) # [layers]

            sim = (v1 * v2).sum(dim=-1) / (norm1 * norm2).clamp(min=1e-8)
            all_sims.append(sim.numpy())

    return np.mean(all_sims, axis=0) # [layers]

def main():
    dir_path = project_path('sentiment/extract_cached')
    files = glob.glob(os.path.join(dir_path, "emotion_style_*.pt"))

    model_groups = defaultdict(list)
    for f in files:
        fname = os.path.basename(f)
        if 'llama3' in fname:
            model_groups['llama3'].append(f)
        elif 'gemma2' in fname:
            model_groups['gemma2'].append(f)
        elif 'qwen2.5' in fname:
            model_groups['qwen2.5'].append(f)

    print("Vector Similarity Analysis (Functional Vectors: Positive - Negative)")
    print("="*80)

    for model_name, paths in model_groups.items():
        print(f"\nModel: {model_name} ({len(paths)} files)")


        if model_name == 'llama3':
            scenario_groups = defaultdict(list)
            for p in paths:
                fname = os.path.basename(p)

                parts = fname.split('_')
                if len(parts) >= 5:
                    scenario = parts[4]
                    scenario_groups[scenario].append(p)

            for scenario, s_paths in scenario_groups.items():
                print(f"  Scenario: {scenario} ({len(s_paths)} files)")
                s_vectors = []
                for p in s_paths:
                    v = get_style_vector(p)
                    if v is not None:
                        s_vectors.append(v)

                if len(s_vectors) >= 2:
                    s_avg_sim = compute_avg_cosine_similarity(s_vectors)
                    print(f"    Avg Similarity: {np.mean(s_avg_sim):.4f}")
                else:
                    print("    Not enough vectors.")

        vectors = []
        for p in paths:
            v = get_style_vector(p)
            if v is not None:
                vectors.append(v)

        if not vectors:
            print("  No valid vectors found.")
            continue

        avg_sim_per_layer = compute_avg_cosine_similarity(vectors)
        if avg_sim_per_layer is not None:
            overall_avg = np.mean(avg_sim_per_layer)
            max_sim_layer = np.argmax(avg_sim_per_layer)
            min_sim_layer = np.argmin(avg_sim_per_layer)

            print(f"  Average Cosine Similarity (Across all layers): {overall_avg:.4f}")
            print(f"  Max Similarity at Layer {max_sim_layer}: {avg_sim_per_layer[max_sim_layer]:.4f}")
            print(f"  Min Similarity at Layer {min_sim_layer}: {avg_sim_per_layer[min_sim_layer]:.4f}")

            # Print a small table of similarity per 5 layers
            print("\n  Layer-wise Similarity (sampled):")
            header = "  Layer | " + " | ".join([f"{i:2d}" for i in range(0, len(avg_sim_per_layer), 5)])
            print(header)
            print("  " + "-" * len(header))
            values = "  Sim   | " + " | ".join([f"{avg_sim_per_layer[i]:.2f}" for i in range(0, len(avg_sim_per_layer), 5)])
            print(values)
        else:
            print("  Not enough vectors to compute similarity.")

if __name__ == "__main__":
    main()
