#!/usr/bin/env python
# -*- coding: utf-8 -*-
from release_paths import project_path

import argparse
import os
import json
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
from matplotlib import font_manager
from itertools import combinations
import warnings
from politeness.geometry import summarize_vectors
warnings.filterwarnings("ignore")


BASE_DIR = project_path('politeness')
parser = argparse.ArgumentParser(description='Plot the seven keigo vectors and summarize their geometry.')
parser.add_argument('--vector-dir', default=os.path.join(BASE_DIR, 'extract_cached'))
parser.add_argument('--output-dir', default=project_path('runs/keigo_geometry/plots'))
parser.add_argument('--reference-index', type=int, choices=range(5), default=1)
args = parser.parse_args()
VEC_DIR, OUT_DIR = args.vector_dir, args.output_dir
os.makedirs(OUT_DIR, exist_ok=True)


_CJK_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(_CJK_FONT_PATH):
    font_manager.fontManager.addfont(_CJK_FONT_PATH)
    _fp = font_manager.FontProperties(fname=_CJK_FONT_PATH)
    CJK = _fp.get_name()
else:
    CJK = "DejaVu Sans"

plt.rcParams.update({
    "font.family": CJK,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})


COLOR = {
    "A": "#2563EB",
    "B": "#DC2626",
    "C": "#16A34A",
}
LABEL = {
    "A": "Shared context (KV cache)",
    "B": "Independent last-token residual",
    "C": "Independent generation-average",
}


FILES_META = []
for i in range(5):
    fname = f"swallow_argmax_detailed_{i}_従業員_to_上司.pt"
    path  = os.path.join(VEC_DIR, fname)
    if os.path.exists(path):
        FILES_META.append({"path": path, "type": "A",
                           "short": f"KV-{i}", "full": f"Shared context v{i}"})

path_lt = os.path.join(VEC_DIR, "swallow_argmax_last_token_0_従業員_to_上司.pt")
if os.path.exists(path_lt):
    FILES_META.append({"path": path_lt, "type": "B",
                       "short": "LastTok-0", "full": "Independent last-token residual v0"})

path_avg = os.path.join(VEC_DIR, "swallow_argmax_twoseq_avg_0_従業員_to_上司.pt")
if os.path.exists(path_avg):
    FILES_META.append({"path": path_avg, "type": "C",
                       "short": "AvgGen-0", "full": "Independent generation-average v0"})

print(f"Loaded {len(FILES_META)} vector files")
for m in FILES_META:
    print(f"  [{m['type']}] {m['short']} <- {os.path.basename(m['path'])}")
if [m['type'] for m in FILES_META] != ['A'] * 5 + ['B', 'C']:
    raise FileNotFoundError('All five JP vectors and both TwoSeq vectors are required')


def load_vector_all_layers(pt_path, residual_type="resid_pre"):
    data = torch.load(pt_path, map_location="cpu", weights_only=False)

    if "difference_vectors" in data:
        v = data["difference_vectors"][residual_type].float()  # [L,H] or [L,1,H]
        if v.dim() == 3:
            v = v.squeeze(1)   # [L, 1, H] → [L, H]
        return v

    r1 = data["generation_1"]["result"]["residuals"][residual_type].float()  # [T1,L,H]
    r2 = data["generation_2"]["result"]["residuals"][residual_type].float()  # [T2,L,H]
    g1 = r1.mean(dim=0)   # [L, H]
    g2 = r2.mean(dim=0)
    return (g2 - g1)      # [L, H]


def load_vector_at_layer(pt_path, layer=15, residual_type="resid_pre"):
    v = load_vector_all_layers(pt_path, residual_type)
    return v[layer]       # [H]


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().flatten(), b.float().flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-9)).item()


all_vecs_all_layers = []   # list of [L, H]
all_vecs_l15        = []   # list of [H]

for m in FILES_META:
    v_all = load_vector_all_layers(m["path"])
    v_l15 = v_all[15]
    all_vecs_all_layers.append(v_all)
    all_vecs_l15.append(v_l15)
    print(f"  {m['short']}: layers={v_all.shape[0]}, H={v_all.shape[1]}, "
          f"L15 norm={v_l15.norm().item():.3f}")

N = len(FILES_META)
num_layers = all_vecs_all_layers[0].shape[0]
types = [m["type"] for m in FILES_META]
shorts = [m["short"] for m in FILES_META]


cos_mat = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        cos_mat[i, j] = cosine_sim(all_vecs_l15[i], all_vecs_l15[j])

fig, ax = plt.subplots(figsize=(7.5, 6))

norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
im = ax.imshow(cos_mat, cmap="RdBu_r", norm=norm, aspect="auto")
plt.colorbar(im, ax=ax, label="Cosine Similarity")

ax.set_xticks(range(N))
ax.set_yticks(range(N))
ax.set_xticklabels(shorts, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(shorts, fontsize=8)

for i in range(N):
    for j in range(N):
        val = cos_mat[i, j]
        color = "white" if abs(val) > 0.5 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=7.5, color=color)


type_changes = []
for k in range(1, N):
    if types[k] != types[k-1]:
        type_changes.append(k - 0.5)
for tc in type_changes:
    ax.axhline(tc, color="black", lw=2)
    ax.axvline(tc, color="black", lw=2)


patches = [mpatches.Patch(color=COLOR[t], label=LABEL[t])
           for t in sorted(set(types))]
ax.legend(handles=patches, loc="upper right", bbox_to_anchor=(1.0, -0.15),
          ncol=1, fontsize=8.5, frameon=True)

ax.set_title("Function-vector cosine similarity matrix (Layer 15, resid_pre)\n"
             "Within-type (blue frame) vs cross-type (outside)", pad=12)


from matplotlib.patches import FancyBboxPatch
type_starts = {}
for k, t in enumerate(types):
    if t not in type_starts:
        type_starts[t] = k
type_ends = {t: max(k for k, tt in enumerate(types) if tt == t) for t in set(types)}
for t in set(types):
    s, e = type_starts[t], type_ends[t]
    rect = plt.Rectangle((s-0.5, s-0.5), e-s+1, e-s+1,
                          linewidth=2.5, edgecolor=COLOR[t], facecolor="none",
                          zorder=5)
    ax.add_patch(rect)

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "fig1_cosine_heatmap.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
print(f"✓ Saved: {out_path}")
plt.close()


vecs_np = np.stack([v.numpy() for v in all_vecs_l15])  # [N, H]

vecs_c = vecs_np - vecs_np.mean(axis=0, keepdims=True)

_, S, Vt = np.linalg.svd(vecs_c, full_matrices=False)
pcs = vecs_c @ Vt[:2].T   # [N, 2]
var_exp = S[:2]**2 / (S**2).sum()

fig, ax = plt.subplots(figsize=(6.5, 5.5))

for i, m in enumerate(FILES_META):
    ax.scatter(pcs[i, 0], pcs[i, 1],
               color=COLOR[m["type"]], s=120, zorder=5,
               edgecolors="white", linewidths=0.8)
    ax.annotate(m["short"], (pcs[i, 0], pcs[i, 1]),
                textcoords="offset points", xytext=(6, 4),
                fontsize=8.5, color=COLOR[m["type"]])


for t in sorted(set(types)):
    idxs = [k for k, tt in enumerate(types) if tt == t]
    if len(idxs) > 1:
        cx, cy = pcs[idxs, 0].mean(), pcs[idxs, 1].mean()
        ax.scatter(cx, cy, marker="*", s=300, color=COLOR[t],
                   edgecolors="black", linewidths=0.8, zorder=6)

patches = [mpatches.Patch(color=COLOR[t], label=LABEL[t]) for t in sorted(set(types))]
ax.legend(handles=patches, fontsize=8.5, loc="best", frameon=True)
ax.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}% var)", fontsize=10)
ax.set_ylabel(f"PC2 ({var_exp[1]*100:.1f}% var)", fontsize=10)
ax.set_title("Function vectors PCA 2D projection (Layer 15)\n(★ = within-type centroid)", pad=10)

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "fig2_pca_scatter.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
print(f"✓ Saved: {out_path}")
plt.close()


cos_by_layer = np.zeros((num_layers, N, N))
for l in range(num_layers):
    for i in range(N):
        for j in range(N):
            vi = all_vecs_all_layers[i][l].float()
            vj = all_vecs_all_layers[j][l].float()
            cos_by_layer[l, i, j] = cosine_sim(vi, vj)


intra_A_pairs  = [(i, j) for i, j in combinations(range(N), 2)
                  if types[i] == "A" and types[j] == "A"]
inter_AB_pairs = [(i, j) for i in range(N) for j in range(N)
                  if types[i] == "A" and types[j] == "B"]
inter_AC_pairs = [(i, j) for i in range(N) for j in range(N)
                  if types[i] == "A" and types[j] == "C"]
inter_BC_pairs = [(i, j) for i in range(N) for j in range(N)
                  if types[i] == "B" and types[j] == "C"]

def mean_cos_at_layers(pair_list, cos_mat_3d):
    if not pair_list:
        return None
    vals = np.array([[cos_mat_3d[l, i, j] for i, j in pair_list]
                     for l in range(cos_mat_3d.shape[0])])  # [L, P]
    return vals.mean(axis=1)  # [L]

intra_A_curve   = mean_cos_at_layers(intra_A_pairs, cos_by_layer)
inter_AB_curve  = mean_cos_at_layers(inter_AB_pairs, cos_by_layer)
inter_AC_curve  = mean_cos_at_layers(inter_AC_pairs, cos_by_layer)
inter_BC_curve  = mean_cos_at_layers(inter_BC_pairs, cos_by_layer)

layers_x = np.arange(num_layers)

fig, ax = plt.subplots(figsize=(9, 5))

if intra_A_curve is not None:
    ax.plot(layers_x, intra_A_curve, color=COLOR["A"], lw=2.5,
            label=f"Intra A-A ({LABEL['A']}, {len(intra_A_pairs)} pairs)")
if inter_AB_curve is not None:
    ax.plot(layers_x, inter_AB_curve, color=COLOR["B"], lw=2, linestyle="--",
            label=f"Cross A-B ({LABEL['A']} vs {LABEL['B']}, {len(inter_AB_pairs)} pairs)")
if inter_AC_curve is not None:
    ax.plot(layers_x, inter_AC_curve, color=COLOR["C"], lw=2, linestyle="-.",
            label=f"Cross A-C ({LABEL['A']} vs {LABEL['C']}, {len(inter_AC_pairs)} pairs)")
if inter_BC_curve is not None:
    ax.plot(layers_x, inter_BC_curve, color="purple", lw=1.5, linestyle=":",
            label=f"Cross B-C ({LABEL['B']} vs {LABEL['C']}, {len(inter_BC_pairs)} pairs)")

ax.axvline(15, color="gray", lw=1.5, linestyle="--", alpha=0.7, label="Layer 15 (injection layer)")
ax.axhline(0, color="black", lw=0.8, linestyle="-", alpha=0.3)

ax.set_xlabel("Transformer Layer", fontsize=10)
ax.set_ylabel("Mean cosine similarity", fontsize=10)
ax.set_title("Layer-wise function-vector cosine similarity: within vs cross type", pad=10)
ax.set_xlim(0, num_layers - 1)
ax.set_ylim(-1.0, 1.05)
ax.legend(fontsize=8.5, loc="lower right", frameon=True)

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "fig3_per_layer_intra_vs_inter.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
print(f"✓ Saved: {out_path}")
plt.close()


L15 = 15
groups = {}
group_labels = []

intra_A_vals = [cos_by_layer[L15, i, j] for i, j in intra_A_pairs]
if intra_A_vals:
    groups["Within\nA-A"] = intra_A_vals
    group_labels.append("A")

inter_AB_vals = [cos_by_layer[L15, i, j] for i, j in inter_AB_pairs]
if inter_AB_vals:
    groups["Cross\nA-B"] = inter_AB_vals
    group_labels.append("B")

inter_AC_vals = [cos_by_layer[L15, i, j] for i, j in inter_AC_pairs]
if inter_AC_vals:
    groups["Cross\nA-C"] = inter_AC_vals
    group_labels.append("C")

inter_BC_vals = [cos_by_layer[L15, i, j] for i, j in inter_BC_pairs]
if inter_BC_vals:
    groups["Cross\nB-C"] = inter_BC_vals
    group_labels.append("BC")

group_colors_map = {"A": COLOR["A"], "B": COLOR["B"], "C": COLOR["C"], "BC": "purple"}

fig, ax = plt.subplots(figsize=(7, 5))

labels_list = list(groups.keys())
data_list   = [groups[k] for k in labels_list]
colors_list = [group_colors_map[gl] for gl in group_labels]

bp = ax.boxplot(data_list, patch_artist=True, notch=False,
                medianprops={"color": "black", "lw": 2},
                whiskerprops={"lw": 1.5}, capprops={"lw": 1.5},
                flierprops={"marker": "o", "markersize": 5})
for patch, col in zip(bp["boxes"], colors_list):
    patch.set_facecolor(col)
    patch.set_alpha(0.5)


for k_idx, (vals, col) in enumerate(zip(data_list, colors_list), start=1):
    jitter = np.random.RandomState(42).uniform(-0.12, 0.12, len(vals))
    ax.scatter([k_idx + j for j in jitter], vals, color=col, s=40,
               alpha=0.8, zorder=5, edgecolors="white", linewidths=0.5)

ax.set_xticks(range(1, len(labels_list) + 1))
ax.set_xticklabels(labels_list, fontsize=9.5)
ax.axhline(0, color="black", lw=0.8, linestyle="--", alpha=0.3)
ax.set_ylabel("Cosine similarity (Layer 15)", fontsize=10)
ax.set_title("Layer 15: within vs cross-type function-vector cosine similarity", pad=10)
ax.set_ylim(-1.0, 1.05)


for k_idx, vals in enumerate(data_list, start=1):
    mu = np.mean(vals)
    ax.text(k_idx, mu + 0.05, f"μ={mu:.2f}", ha="center", fontsize=8,
            color="black")

patches = [mpatches.Patch(color=COLOR[t], label=LABEL[t]) for t in sorted(set(types))]
ax.legend(handles=patches, fontsize=8.5, loc="lower right", frameon=True)

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "fig4_distribution_boxplot.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
print(f"✓ Saved: {out_path}")
plt.close()


key_layers = [0, 4, 8, 12, 15, 20, 24, 28, 31]

fig, axes = plt.subplots(3, 3, figsize=(13, 11))
axes_flat = axes.flatten()

for ax_i, l in enumerate(key_layers):
    ax = axes_flat[ax_i]
    mat = cos_by_layer[l]
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    im = ax.imshow(mat, cmap="RdBu_r", norm=norm, aspect="auto")
    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xticklabels(shorts, rotation=40, ha="right", fontsize=6.5)
    ax.set_yticklabels(shorts, fontsize=6.5)
    for ii in range(N):
        for jj in range(N):
            v = mat[ii, jj]
            c = "white" if abs(v) > 0.55 else "black"
            ax.text(jj, ii, f"{v:.2f}", ha="center", va="center",
                    fontsize=5.5, color=c)

    for tc in type_changes:
        ax.axhline(tc, color="black", lw=1.2)
        ax.axvline(tc, color="black", lw=1.2)

    for t in set(types):
        s_idx, e_idx = type_starts[t], type_ends[t]
        rect = plt.Rectangle((s_idx-0.5, s_idx-0.5), e_idx-s_idx+1, e_idx-s_idx+1,
                              lw=1.8, edgecolor=COLOR[t], facecolor="none", zorder=5)
        ax.add_patch(rect)
    # highlight layer 15
    title_color = "red" if l == 15 else "black"
    ax.set_title(f"Layer {l}", fontsize=9, color=title_color,
                 fontweight="bold" if l == 15 else "normal")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.suptitle("Transformer-layer cosine similarity matrices\n(layer 15 title in red marks the injection layer)",
             fontsize=12, y=1.01)
patches = [mpatches.Patch(color=COLOR[t], label=LABEL[t]) for t in sorted(set(types))]
fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=9,
           bbox_to_anchor=(0.5, -0.03), frameon=True)
plt.tight_layout()
out_path = os.path.join(OUT_DIR, "fig5_multilayer_heatmaps.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
print(f"✓ Saved: {out_path}")
plt.close()


def stats(vals):
    if not vals:
        return {}
    arr = np.array(vals)
    return {"n": len(arr), "mean": float(arr.mean()), "std": float(arr.std()),
            "min": float(arr.min()), "max": float(arr.max())}

summary = {
    "geometry_scope": summarize_vectors(all_vecs_l15[:5], all_vecs_l15[5], all_vecs_l15[6], args.reference_index),
    "layer_15_analysis": {
        "intra_A_A (within shared context)": stats(intra_A_vals),
        "inter_A_B (shared context vs last-token residual)": stats(inter_AB_vals),
        "inter_A_C (shared context vs generation-average)": stats(inter_AC_vals),
        "inter_B_C (last-token residual vs generation-average)": stats(inter_BC_vals),
    },
    "per_layer_mean_cosine": {
        "intra_A": intra_A_curve.tolist() if intra_A_curve is not None else [],
        "inter_AB": inter_AB_curve.tolist() if inter_AB_curve is not None else [],
        "inter_AC": inter_AC_curve.tolist() if inter_AC_curve is not None else [],
    },
    "vector_files": [{"short": m["short"], "type": m["type"],
                      "file": os.path.basename(m["path"]),
                      "L15_norm": float(all_vecs_l15[k].norm().item())}
                     for k, m in enumerate(FILES_META)],
}

out_json = os.path.join(OUT_DIR, "cluster_analysis_result.json")
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f"✓ Saved: {out_json}")


print("\n" + "="*65)
print("  Function vector clustering summary (Layer 15, resid_pre)")
print("="*65)
for group_name, stat in summary["layer_15_analysis"].items():
    if stat:
        print(f"\n{group_name}:")
        print(f"  n={stat['n']}, mean={stat['mean']:.4f}, "
              f"std={stat['std']:.4f}, "
              f"[{stat['min']:.4f}, {stat['max']:.4f}]")
print("\n" + "="*65)
if intra_A_curve is not None and inter_AB_curve is not None:
    diff = intra_A_curve[15] - inter_AB_curve[15]
    print(f"  Layer 15 gap (A intra - A-B cross) = {diff:.4f}")
print("="*65)
print(f"\nAll figures saved to: {OUT_DIR}")
