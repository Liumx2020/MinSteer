from release_paths import project_path
import torch
import os
import glob

def average_vectors(n_samples, vector_dir=None):

    if vector_dir is None:
        vector_dir = project_path('runs/sentiment_scaling/raw')
    files = glob.glob(os.path.join(vector_dir, "emotion_style_llama3_en_*_negative_argmax_sample_*.pt"))
    files = sorted(files)

    if len(files) < n_samples:
        print(f"Warning: Only {len(files)} files found, requested {n_samples}")
        n_samples = len(files)

    selected_files = files[:n_samples]

    all_layer_vecs = []

    for fp in selected_files:
        try:
            data = torch.load(fp, map_location="cpu")
            # residuals shape: [tokens, layers, hidden_dim]
            res1 = data["generation_1"]["result"]["residuals"]["resid_post"]
            res2 = data["generation_2"]["result"]["residuals"]["resid_post"]

            # Average over tokens for each layer
            avg1 = res1.mean(dim=0) # [layers, hidden_dim]
            avg2 = res2.mean(dim=0) # [layers, hidden_dim]

            vec = avg2 - avg1
            all_layer_vecs.append(vec)
        except Exception as e:
            print(f"Error processing {fp}: {e}")

    if not all_layer_vecs:
        return None


    final_vec = torch.stack(all_layer_vecs).mean(dim=0) # [layers, hidden_dim]
    return final_vec

def save_scaling_vectors():
    output_dir = project_path('runs/sentiment_scaling/vectors')
    os.makedirs(output_dir, exist_ok=True)

    for n in [1, 5, 10, 50]:
        print(f"Calculating for n={n}...")
        vec = average_vectors(n)
        if vec is not None:
            # Save the full [32, 4096] tensor
            out_path = os.path.join(output_dir, f"minsteer_vec_n{n}.pt")
            torch.save(vec, out_path)
            print(f"Saved to {out_path}")

            # Also save individual layers in CAA format for easier testing
            for layer in range(vec.size(0)):
                layer_dir = os.path.join(output_dir, f"n{n}")
                os.makedirs(layer_dir, exist_ok=True)
                layer_path = os.path.join(layer_dir, f"vec_layer_{layer}_Meta-Llama-3.1-8B-Instruct.pt")
                torch.save(vec[layer], layer_path)

if __name__ == "__main__":
    save_scaling_vectors()
