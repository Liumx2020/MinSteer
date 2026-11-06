#!/usr/bin/env python
# -*- coding: utf-8 -*-
from release_paths import project_path

import os
import sys
import json
import glob
import argparse
import datetime
import importlib.util
import torch
import torch.nn.functional as F

BASE_DIR = project_path('politeness')
JP_DIR   = os.path.join(BASE_DIR, "extract_cached")
EN_DIR   = os.path.join(BASE_DIR, "vectors_english")
OUT_DIR  = os.path.join(BASE_DIR, "analyze_pca")
os.makedirs(OUT_DIR, exist_ok=True)

NUM_LAYERS  = 32
HIDDEN_SIZE = 4096


def load_jp_vector(pt_path, layer, residual_type="resid_pre"):
    data = torch.load(pt_path, map_location="cpu")
    r1 = data["generation_1"]["result"]["residuals"][residual_type]  # [T, L, H]
    r2 = data["generation_2"]["result"]["residuals"][residual_type]
    g1 = r1[:, layer, :].float().mean(dim=0)
    g2 = r2[:, layer, :].float().mean(dim=0)
    return g2 - g1


def load_en_vector(pt_path, layer, residual_type="resid_pre"):
    data = torch.load(pt_path, map_location="cpu")
    vec = data.get("vectors", {}).get(residual_type)
    if vec is None:
        return None
    v = vec[layer, :].float() if vec.dim() == 2 else vec.float()
    # Saved JP, EN and CN directions all use casual minus formal.
    return v


def load_cn_vector(pt_path, layer, residual_type="resid_pre"):
    return load_en_vector(pt_path, layer, residual_type)


def cosine_sim(a, b):
    a = F.normalize(a.flatten().float().unsqueeze(0), dim=1)
    b = F.normalize(b.flatten().float().unsqueeze(0), dim=1)
    return (a @ b.T).item()


CN_FORMAL_TEXTS = [
    "请问您是否方便在近期安排一次会议，以便我们就相关事宜进行深入探讨？",
    "敬请您在百忙之中拨冗审阅附件材料，感激不尽。",
    "如有任何疑问，敬请随时与我联系，我将竭诚为您解答。",
    "感谢您对本项目的支持与配合，期待与您进一步合作。",
    "恳请您不吝赐教，对本方案提出宝贵意见。",
]

CN_CASUAL_TEXTS = [
    "你最近有空开个会聊聊吗？",
    "能不能帮我看一下这份文件？",
    "有问题的话随时找我说一声。",
    "谢了哈，咱们后续继续合作。",
    "你觉得这个方案怎么样？有啥意见说说吧。",
]


def extract_cn_vector(model_name="swallow", layer=15, residual_type="resid_pre"):
    model_paths = {
        "swallow": "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3",
        "llama_en": "meta-llama/Llama-3.1-8B-Instruct",
    }
    model_path = model_paths.get(model_name, model_paths["swallow"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    script3_path = os.path.join(BASE_DIR, "extract_cached.py")
    spec = importlib.util.spec_from_file_location("residual_extract", script3_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    extract_fn = mod.token_wise_generation_with_residuals

    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"[CN] 加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    eot_id = tokenizer.eos_token_id

    formal_resids, casual_resids = [], []
    pairs = list(zip(CN_FORMAL_TEXTS, CN_CASUAL_TEXTS))
    print(f"[CN] 提取 {len(pairs)} 对中文句子的残差流 (layer={layer})")
    for i, (f_text, c_text) in enumerate(pairs):
        for style, text in [("formal", f_text), ("casual", c_text)]:
            messages = [{"role": "user", "content": text}]
            prompt   = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            input_ids = tokenizer(prompt, return_tensors="pt").to(device)["input_ids"]
            try:
                out = extract_fn(model, tokenizer, input_ids=input_ids, max_new_tokens=5,
                                 sampling_method="argmax", force_append_eot_id=eot_id)
                r = out["residuals"].get(residual_type)
                if r is not None:
                    (formal_resids if style == "formal" else casual_resids).append(r)
                print(f"  [{i+1}] {style}: OK (len={r.shape[0] if r is not None else '?'})")
            except Exception as e:
                print(f"  [{i+1}] {style}: 失败 - {e}")
        torch.cuda.empty_cache()

    n = min(len(formal_resids), len(casual_resids))
    if n == 0:
        print("[CN] ❌ 无有效样本，跳过")
        return None


    diffs = []
    for i in range(n):
        f = formal_resids[i]   # [T, L, H]
        c = casual_resids[i]
        min_t = min(f.shape[0], c.shape[0])
        diffs.append(c[:min_t].float().mean(0) - f[:min_t].float().mean(0))  # [L, H]
    vec = torch.stack(diffs).mean(0)  # [L, H]

    out_file = os.path.join(EN_DIR, f"chinese_formal_casual_{model_name}_layer{layer}.pt")
    torch.save({
        "model_name": model_name, "language": "chinese",
        "direction": "casual_minus_formal", "layer": layer,
        "residual_type": residual_type, "vectors": {residual_type: vec},
        "n_prompts": n, "timestamp": datetime.datetime.now().isoformat(),
    }, out_file)
    print(f"[CN] 💾 已保存: {out_file}")
    del model
    torch.cuda.empty_cache()
    return out_file


def layer_sweep(named_vectors, residual_type="resid_pre", ref_name=None):
    names  = list(named_vectors.keys())
    vecs   = {n: [] for n in names}   # layer → [H]
    for layer in range(NUM_LAYERS):
        for n, fn in named_vectors.items():
            try:
                v = fn(layer)
                vecs[n].append(v)
            except Exception:
                vecs[n].append(None)

    results = {}

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            na, nb = names[i], names[j]
            sims = []
            for layer in range(NUM_LAYERS):
                va, vb = vecs[na][layer], vecs[nb][layer]
                if va is not None and vb is not None:
                    sims.append(round(cosine_sim(va, vb), 5))
                else:
                    sims.append(None)
            key = f"{na}__vs__{nb}"
            results[key] = sims
    return results


def pca_analysis(named_vectors_at_layer):
    """
    named_vectors_at_layer: dict {name: tensor [H]}
    Fit PC1 to mean-centered unit vectors. Report raw input vectors projected
    onto that axis, rather than centered PCA scores, as in the paper's panel B.
    Returns: dict with PCA results
    """
    names = list(named_vectors_at_layer.keys())
    vecs  = [F.normalize(named_vectors_at_layer[n].float().flatten(), dim=0)
             for n in names]
    M = torch.stack(vecs)  # [N, H]


    mean = M.mean(0)
    Mc   = M - mean


    _, S, Vt = torch.linalg.svd(Mc, full_matrices=False)
    var_total    = (S ** 2).sum().item()
    var_pc1      = (S[0] ** 2).item()
    var_pc1_ratio = var_pc1 / var_total if var_total > 0 else 0


    pc1          = Vt[0]  # [H]
    projections  = {n: (named_vectors_at_layer[n].float().flatten() @ pc1).item()
                    for n in names}
    signs        = [1 if v > 0 else -1 for v in projections.values()]
    sign_agree   = all(s == signs[0] for s in signs)


    pairwise = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pairwise[f"{names[i]}_{names[j]}"] = round(cosine_sim(vecs[i], vecs[j]), 5)

    return {
        "n_vectors": len(names),
        "names": names,
        "variance_explained_PC1": round(var_pc1_ratio, 4),
        "projections_on_PC1": {k: round(v, 5) for k, v in projections.items()},
        "sign_consistent": sign_agree,
        "pairwise_cosine": pairwise,
        "singular_values": [round(s.item(), 4) for s in S[:min(5, len(S))]],
    }


def plot_layer_sweep(sweep_results, out_dir, tag=""):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 5))
        colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]
        for idx, (pair, sims) in enumerate(sweep_results.items()):
            layers = list(range(len(sims)))
            vals   = [s if s is not None else float("nan") for s in sims]
            ax.plot(layers, vals, label=pair.replace("__vs__", " vs "), color=colors[idx % len(colors)], linewidth=1.5)

        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.axhline(0.5, color="orange", linewidth=0.8, linestyle=":", label="threshold 0.5")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Cosine Similarity (casual minus formal)")
        ax.set_title("Layer-wise Cosine Similarity: JP vs EN Register Vectors")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(out_dir, f"layer_sweep{tag}.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"📊 已保存曲线图: {path}")
    except ImportError:
        print("⚠ matplotlib 不可用，跳过绘图")


def plot_pairwise_heatmap(pca_result, out_dir, tag=""):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        names = pca_result["names"]
        n     = len(names)
        mat   = np.zeros((n, n))
        for i in range(n):
            mat[i, i] = 1.0
            for j in range(i + 1, n):
                key = f"{names[i]}_{names[j]}"
                v   = pca_result["pairwise_cosine"].get(key, 0.0)
                mat[i, j] = v
                mat[j, i] = v

        fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
        ax.set_yticklabels(names, fontsize=9)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(mat[i, j]) > 0.5 else "black")
        plt.colorbar(im, ax=ax)
        ax.set_title(f"Pairwise Cosine Similarity (casual minus formal)\nPC1 variance={pca_result['variance_explained_PC1']:.1%}, sign_consistent={pca_result['sign_consistent']}")
        plt.tight_layout()
        path = os.path.join(out_dir, f"pairwise_heatmap{tag}.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"📊 已保存热力图: {path}")
    except ImportError:
        print("⚠ matplotlib 不可用，跳过绘图")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=15, help='Zero-based layer index.')
    parser.add_argument("--residual_type", type=str, default="resid_pre",
                        choices=["resid_pre", "resid_mid", "resid_post"])
    parser.add_argument("--extract_cn", action="store_true",
                        help='Extract cn.')
    parser.add_argument("--cn_model", type=str, default="swallow", choices=["swallow", "llama_en"])
    parser.add_argument("--jp_indices", type=str, default="0,1,2,3,4",
                        help='Jp indices.')
    parser.add_argument("--no_layer_sweep", action="store_true", help='No layer sweep.')
    args = parser.parse_args()

    rt    = args.residual_type
    layer = args.layer


    jp_indices = [int(i) for i in args.jp_indices.split(",")]
    jp_files   = {}
    for idx in jp_indices:
        pattern = os.path.join(JP_DIR, f"swallow_argmax_detailed_{idx}_従業員_to_上司.pt")
        if os.path.exists(pattern):
            jp_files[f"JP_v{idx}"] = pattern
    if not jp_files:
        print("❌ 未找到日语向量文件，请先运行脚本 3")
        sys.exit(1)

    en_swallow_path = os.path.join(EN_DIR, "english_formal_casual_swallow_layer15.pt")
    en_llama_path   = os.path.join(EN_DIR, "english_formal_casual_llama_en_layer15.pt")

    print(f"[INFO] 日语向量: {list(jp_files.keys())}")
    print(f"[INFO] EN-Swallow: {'✅' if os.path.exists(en_swallow_path) else '❌'}")
    print(f"[INFO] EN-LLaMA:   {'✅' if os.path.exists(en_llama_path) else '❌'}")


    cn_path = os.path.join(EN_DIR, f"chinese_formal_casual_{args.cn_model}_layer{layer}.pt")
    if args.extract_cn or not os.path.exists(cn_path):
        if args.extract_cn:
            cn_path = extract_cn_vector(args.cn_model, layer, rt) or cn_path
        else:
            print(f"[INFO] CN 向量不存在，跳过（可用 --extract_cn 生成）")
            cn_path = None
    print(f"[INFO] CN-{args.cn_model}: {'✅' if cn_path and os.path.exists(cn_path) else '❌ (跳过)'}")


    named_loaders = {}
    for name, path in jp_files.items():
        named_loaders[name] = (lambda p: lambda l: load_jp_vector(p, l, rt))(path)

    if os.path.exists(en_swallow_path):
        named_loaders["EN_Swallow"] = lambda l: load_en_vector(en_swallow_path, l, rt)
    if os.path.exists(en_llama_path):
        named_loaders["EN_LLaMA"]   = lambda l: load_en_vector(en_llama_path, l, rt)
    if cn_path and os.path.exists(cn_path):
        named_loaders["CN_Swallow"] = lambda l: load_cn_vector(cn_path, l, rt)

    print(f"\n[INFO] 共 {len(named_loaders)} 个向量参与分析: {list(named_loaders.keys())}")


    sweep_results = {}
    if not args.no_layer_sweep:
        print(f"\n⏳ 逐层扫描（{NUM_LAYERS} 层 × {len(named_loaders)} 向量）...")
        sweep_results = layer_sweep(named_loaders, rt)
        for pair, sims in sweep_results.items():
            valid = [s for s in sims if s is not None]
            peak_layer = sims.index(max(valid)) if valid else -1
            peak_val   = max(valid) if valid else 0
            print(f"  {pair}: peak_layer={peak_layer}, peak_cos={peak_val:.4f}")
        plot_layer_sweep(sweep_results, OUT_DIR, f"_layer{layer}")


    print(f"\n⏳ 计算 layer={layer} 的多向量 PCA ...")
    vecs_at_layer = {}
    for name, fn in named_loaders.items():
        try:
            v = fn(layer)
            if v is not None:
                vecs_at_layer[name] = v
        except Exception as e:
            print(f"  ⚠ {name} 加载失败: {e}")

    if len(vecs_at_layer) < 2:
        print("❌ 有效向量不足 2 个，无法做 PCA")
        sys.exit(1)

    pca_result = pca_analysis(vecs_at_layer)
    print(f"\n{'='*55}")
    print(f"  PCA 分析结果 (layer={layer}, {rt})")
    print(f"{'='*55}")
    print(f"  向量数量       : {pca_result['n_vectors']}")
    print(f"  PC1 方差解释率 : {pca_result['variance_explained_PC1']:.1%}")
    print(f"  方向符号一致   : {'✅ YES' if pca_result['sign_consistent'] else '❌ NO'}")
    print(f"  奇异值         : {pca_result['singular_values']}")
    print(f"\n  PC1 投影:")
    for name, val in pca_result["projections_on_PC1"].items():
        bar = "+" * int(abs(val) * 20) if val > 0 else "-" * int(abs(val) * 20)
        print(f"    {name:20s}: {val:+.4f}  {bar}")
    print(f"\n  成对余弦相似度:")
    for pair, sim in pca_result["pairwise_cosine"].items():
        flag = "⭐" if abs(sim) > 0.5 else ("✓" if abs(sim) > 0.3 else "·")
        print(f"    {pair:40s}: {sim:+.4f}  {flag}")
    print(f"{'='*55}")


    pc1_var = pca_result["variance_explained_PC1"]
    sign_ok = pca_result["sign_consistent"]
    avg_jp_en_cos = []
    for pair, sim in pca_result["pairwise_cosine"].items():
        if ("JP_" in pair and "EN_" in pair) or ("JP_" in pair and "CN_" in pair):
            avg_jp_en_cos.append(sim)
    avg_cross = sum(avg_jp_en_cos) / len(avg_jp_en_cos) if avg_jp_en_cos else 0

    if pc1_var > 0.5 and sign_ok:
        interpretation = "强支持「语言无关 register 子空间」假说：PC1 解释大部分方差，且各语言方向投影符号一致。"
    elif pc1_var > 0.3 or (sign_ok and avg_cross > 0.2):
        interpretation = "中等支持：存在共享分量但不主导。跨语言迁移依赖部分对齐的共享子空间，还有语言特异噪音。"
    else:
        interpretation = "弱支持：各语言向量分散，register 可能更多编码为语言特异方向，跨语言效果需其他机制解释。"
    print(f"\n💡 解读：{interpretation}")


    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result = {
        "timestamp": ts,
        "config": {"layer": layer, "residual_type": rt, "n_jp": len(jp_files)},
        "pca": pca_result,
        "layer_sweep_peak": {
            pair: {
                "peak_layer": int(sims.index(max(s for s in sims if s is not None)))
                    if any(s is not None for s in sims) else -1,
                "peak_cos": round(max((s for s in sims if s is not None), default=0), 5),
                "layer15_cos": sims[15] if len(sims) > 15 else None,
            }
            for pair, sims in sweep_results.items()
        } if sweep_results else {},
        "interpretation": interpretation,
    }
    out_path = os.path.join(OUT_DIR, f"pca_result_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n💾 已保存: {out_path}")

    plot_pairwise_heatmap(pca_result, OUT_DIR, f"_layer{layer}_{ts}")


if __name__ == "__main__":
    main()
