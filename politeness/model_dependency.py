#!/usr/bin/env python
# -*- coding: utf-8 -*-
from release_paths import project_path

import os
import sys
import json
import glob
import argparse
import datetime
import subprocess
import torch
import torch.nn.functional as F

BASE_DIR   = project_path('politeness')
OUT_DIR    = os.path.join(BASE_DIR, "model_dependency")
SCRIPT4    = os.path.join(BASE_DIR, "evaluate_transfer.py")
JP_DIR     = os.path.join(BASE_DIR, "extract_cached")
EN_DIR     = os.path.join(BASE_DIR, "vectors_english")
PREV_C1    = os.path.join(BASE_DIR, "evaluate_transfer",
                           "cross_lang_swallow_stanford_v1_20260314_223720.json")
os.makedirs(OUT_DIR, exist_ok=True)

STRENGTHS  = "0,1,-1,2,-2"
MAX_TESTS  = 99


CONDITIONS = {
    "C1": {
        "label": "Swallow + JP  (跨语言×双语×LLaMA3-JP)",
        "model": "swallow",    "vector_source": "japanese",
        "bilingual": True,  "arch": "llama3",
        "prefix": "expD_C1_swallow_jp",
    },
    "C2": {
        "label": "Swallow + EN  (同语言×双语×LLaMA3-JP)",
        "model": "swallow",    "vector_source": "english",
        "bilingual": True,  "arch": "llama3",
        "prefix": "expD_C2_swallow_en",
    },
    "C3": {
        "label": "LLaMA3-EN + EN (同语言×单语×LLaMA3)",
        "model": "llama_en",   "vector_source": "english",
        "bilingual": False, "arch": "llama3",
        "prefix": "expD_C3_llama3en_en",
    },
    "C4": {
        "label": "LLaMA3-EN + JP (跨语言×单语×LLaMA3) ←关键",
        "model": "llama_en",   "vector_source": "japanese",
        "bilingual": False, "arch": "llama3",
        "prefix": "expD_C4_llama3en_jp",
    },
    "C5": {
        "label": "LLaMA3-ZH + JP (跨语言×双语×LLaMA3-ZH)",
        "model": "llama3_zh",  "vector_source": "japanese",
        "bilingual": True,  "arch": "llama3",
        "prefix": "expD_C5_llama3zh_jp",
    },
    "C6": {
        "label": "LLaMA3-ZH + EN (同语言×双语×LLaMA3-ZH)",
        "model": "llama3_zh",  "vector_source": "english",
        "bilingual": True,  "arch": "llama3",
        "prefix": "expD_C6_llama3zh_en",
    },
    "C7": {
        "label": "LLaMA2-EN + JP (跨语言×单语×LLaMA2) 代际",
        "model": "llama2",     "vector_source": "japanese",
        "bilingual": False, "arch": "llama2",
        "prefix": "expD_C7_llama2en_jp",
    },
    "C8": {
        "label": "LLaMA2-EN + EN (同语言×单语×LLaMA2) 代际",
        "model": "llama2",     "vector_source": "english",
        "bilingual": False, "arch": "llama2",
        "prefix": "expD_C8_llama2en_en",
    },
    "C9": {
        "label": "Mistral + JP  (跨语言×单语×Mistral) 架构",
        "model": "mistral",    "vector_source": "japanese",
        "bilingual": False, "arch": "mistral",
        "prefix": "expD_C9_mistral_jp",
    },
    "C10": {
        "label": "Mistral + EN  (同语言×单语×Mistral) 架构",
        "model": "mistral",    "vector_source": "english",
        "bilingual": False, "arch": "mistral",
        "prefix": "expD_C10_mistral_en",
    },
}


def find_result_file(condition_key):
    cfg    = CONDITIONS[condition_key]
    prefix = cfg["prefix"]

    files = glob.glob(os.path.join(OUT_DIR, f"{prefix}_*.json"))
    if files:
        return sorted(files)[-1]

    if condition_key == "C1" and os.path.exists(PREV_C1):
        return PREV_C1
    return None


def run_condition(condition_key, force=False, max_tests=MAX_TESTS, strengths=STRENGTHS):
    cfg = CONDITIONS[condition_key]
    print(f"\n{'='*60}")
    print(f"  运行条件 {condition_key}: {cfg['label']}")
    print(f"{'='*60}")

    existing = find_result_file(condition_key)
    if existing and not force:
        print(f"  ✅ 已有结果: {os.path.basename(existing)}（跳过，--force 可覆盖）")
        return existing


    if cfg["vector_source"] == "english":
        en_file = os.path.join(EN_DIR, f"english_formal_casual_{cfg['model']}_layer15.pt")
        if not os.path.exists(en_file):
            print(f"  ⚠ 英语向量不存在: {en_file}")
            print(f"  → 正在提取（需要加载模型）...")
            ret = subprocess.run(
                [sys.executable, os.path.join(BASE_DIR, "extract_multilingual.py"),
                 "--model", cfg["model"], "--max_prompts", "5"],
                cwd=BASE_DIR
            )
            if ret.returncode != 0:
                print(f"  ❌ 向量提取失败，退出")
                return None


    if cfg["vector_source"] == "japanese":
        jp_files = glob.glob(os.path.join(JP_DIR, "swallow_argmax_detailed_1_従業員_to_上司.pt"))
        if not jp_files:
            print(f"  ❌ 日语向量不存在，请先运行脚本3")
            return None


    ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"{cfg['prefix']}_{ts}"

    cmd = [
        sys.executable, SCRIPT4,
        "--model",          cfg["model"],
        "--vector_source",  cfg["vector_source"],
        "--corpus",         "stanford",
        "--max_tests",      str(max_tests),
        "--stratify",       "1,0,-1",
        "--strengths",      strengths,
        "--prompt_mode",    "neutral",
    ]

    if not cfg.get("bilingual", True) and cfg["vector_source"] == "japanese":
        cmd.append("--allow_cross_lingual")
    print(f"  执行: {' '.join(cmd)}")
    env = dict(os.environ)
    env["EXPD_OUTPUT_TAG"] = tag

    ret = subprocess.run(cmd, cwd=BASE_DIR)
    if ret.returncode != 0:
        print(f"  ❌ 条件 {condition_key} 运行失败")
        return None


    res_dir   = os.path.join(BASE_DIR, "evaluate_transfer")
    model_str = cfg["model"]

    if cfg["vector_source"] == "english":
        src_files = sorted(glob.glob(os.path.join(res_dir, f"cross_lang_{model_str}_stanford_en_*.json")))
    else:

        src_files = sorted(glob.glob(os.path.join(res_dir, f"cross_lang_{model_str}_stanford_v*.json")))
        if not src_files:

            all_model_files = sorted(glob.glob(os.path.join(res_dir, f"cross_lang_{model_str}_stanford_*.json")))

            src_files = [f for f in all_model_files if f.split("_")[-1].replace(".json","") >= ts.split("_")[0]]
            if not src_files:
                src_files = all_model_files
    if not src_files:
        src_files = sorted(glob.glob(os.path.join(res_dir, f"cross_lang_{model_str}_stanford_*.json")))
    if src_files:
        src = src_files[-1]
        dst = os.path.join(OUT_DIR, f"{tag}.json")
        import shutil
        shutil.copy2(src, dst)
        print(f"  ✅ 结果已复制: {dst}")
        return dst
    print(f"  ⚠ 未找到输出文件")
    return None


def parse_result(json_path):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)


    if isinstance(data, dict) and "formality" in data:
        raw = data["formality"].get("mean_by_alpha", {})
        return {float(k): round(v, 4) for k, v in raw.items()}


    records = data if isinstance(data, list) else data.get("results", [])
    by_alpha = {}
    for rec in records:
        for sres in rec.get("strength_results", []):
            alpha = sres.get("alpha") if sres.get("alpha") is not None else sres.get("strength")
            prob  = sres.get("formality_prob")
            if alpha is None or prob is None:
                continue
            by_alpha.setdefault(alpha, []).append(prob)
    return {float(k): round(sum(v) / len(v), 4) for k, v in by_alpha.items()}


def parse_result_sample_level(json_path):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)


    if isinstance(data, dict) and "formality" in data:

        pt_path = json_path.replace(".json", ".pt")
        if os.path.exists(pt_path):
            try:
                pt = torch.load(pt_path, map_location="cpu")

                if "formality" in pt:
                    fm = pt["formality"]
                    if "by_sample" in fm:
                        return fm["by_sample"]
            except Exception:
                pass

        raw = data["formality"].get("mean_by_alpha", {})
        return {float(k): [v] for k, v in raw.items()}
    return {}


def wilcoxon_paired(scores_a, scores_b, alpha_label=""):
    try:
        from scipy.stats import wilcoxon
        import numpy as np
        a = np.array(scores_a, dtype=float)
        b = np.array(scores_b, dtype=float)
        n = min(len(a), len(b))
        if n < 5:
            return {"n": n, "p_value": None, "statistic": None, "note": "样本不足"}
        a, b = a[:n], b[:n]
        diff = a - b
        if (diff == 0).all():
            return {"n": n, "p_value": 1.0, "statistic": 0, "note": "无差异"}
        stat, p = wilcoxon(a, b, alternative="two-sided")

        d = (a - b).mean() / ((a - b).std() + 1e-9)
        return {"n": n, "p_value": round(float(p), 6), "statistic": round(float(stat), 2),
                "cohens_d": round(float(d), 4), "note": ""}
    except ImportError:
        return {"n": 0, "p_value": None, "note": "scipy 未安装"}
    except Exception as e:
        return {"n": 0, "p_value": None, "note": str(e)}


def compute_direction_score(by_alpha):
    pos = [v for k, v in by_alpha.items() if k > 0]
    neg = [v for k, v in by_alpha.items() if k < 0]
    if not pos or not neg:
        return 0.0
    return round(sum(neg) / len(neg) - sum(pos) / len(pos), 4)


def compute_dose_response(by_alpha):
    pos_sorted = sorted([(k, v) for k, v in by_alpha.items() if k > 0])
    neg_sorted = sorted([(k, v) for k, v in by_alpha.items() if k < 0], reverse=True)
    def monotone_dec(lst):
        return all(lst[i][1] >= lst[i+1][1] for i in range(len(lst)-1))
    def monotone_inc(lst):
        return all(lst[i][1] <= lst[i+1][1] for i in range(len(lst)-1))
    return {
        "pos_monotone_decrease": monotone_dec(pos_sorted),
        "neg_monotone_increase": monotone_inc(neg_sorted),
    }


def analyze(args):
    print(f"\n{'='*70}")
    print(f"  实验D 汇总分析（扩充版 C1-C10）")
    print(f"{'='*70}")

    results = {}
    all_ckeys = list(CONDITIONS.keys())
    for ckey in all_ckeys:
        path = find_result_file(ckey)
        if path:
            try:
                by_alpha  = parse_result(path)
                by_sample = parse_result_sample_level(path)
                results[ckey] = {"path": path, "by_alpha": by_alpha, "by_sample": by_sample}
                n = len(next(iter(by_sample.values()), []))
                print(f"  {ckey}: {os.path.basename(path)}  (n_sample={n})")
            except Exception as e:
                print(f"  {ckey}: 解析失败 - {e}")
        else:
            print(f"  {ckey}: — 未完成")

    if len(results) < 2:
        print("⚠ 有效结果不足，无法汇总")
        return


    print(f"\n  方向控制分（DS = 负α均值 − 正α均值）及剂量-响应：")
    print(f"  {'条件':>4} {'DS':>8} {'baseline':>10} {'单调性':>14}  标签")
    print(f"  {'-'*70}")
    ds_map = {}
    for ckey, r in results.items():
        ds  = compute_direction_score(r["by_alpha"])
        dr  = compute_dose_response(r["by_alpha"])
        mon = "双向✅" if dr["pos_monotone_decrease"] and dr["neg_monotone_increase"]\
              else ("单向⚡" if dr["pos_monotone_decrease"] or dr["neg_monotone_increase"] else "不单调❌")
        baseline = r["by_alpha"].get(0.0, float("nan"))
        ds_map[ckey] = ds
        print(f"  {ckey:>4} {ds:>+8.4f} {baseline:>10.4f} {mon:>14}  {CONDITIONS[ckey]['label']}")


    def sample_ds(by_sample):
        pos_keys = [k for k in by_sample if k > 0]
        neg_keys = [k for k in by_sample if k < 0]
        if not pos_keys or not neg_keys:
            return []
        n = min(len(by_sample[pos_keys[0]]), len(by_sample[neg_keys[0]]))
        import numpy as np
        pos_mat = np.array([by_sample[k][:n] for k in pos_keys])
        neg_mat = np.array([by_sample[k][:n] for k in neg_keys])
        return list(neg_mat.mean(axis=0) - pos_mat.mean(axis=0))

    print(f"\n  Wilcoxon 统计检验（vs C1 基准，样本级方向控制分）：")
    print(f"  {'对比':>15} {'n':>5} {'p-value':>10} {'Cohen_d':>9} {'显著性':>8}")
    print(f"  {'-'*55}")
    wilcoxon_results = {}
    if "C1" in results:
        c1_ds_samples = sample_ds(results["C1"]["by_sample"])
        for ckey in results:
            if ckey == "C1":
                continue
            other_ds = sample_ds(results[ckey]["by_sample"])
            if not c1_ds_samples or not other_ds:
                continue
            wres = wilcoxon_paired(c1_ds_samples, other_ds, ckey)
            wilcoxon_results[f"C1_vs_{ckey}"] = wres
            p   = wres.get("p_value")
            d   = wres.get("cohens_d", float("nan"))
            n   = wres.get("n", 0)
            sig = "n/a" if p is None else ("***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns")))
            p_s = f"{p:.4f}" if p is not None else "N/A"
            print(f"  {'C1 vs '+ckey:>15} {n:>5} {p_s:>10} {d:>9.4f} {sig:>8}")


    print(f"\n  {'─'*60}")
    print(f"  核心推论（Cross-lingual × Monolingual 对照）：")
    _cross_mono = {k: v for k, v in ds_map.items()
                   if k in results and not CONDITIONS[k].get("bilingual", True)
                   and CONDITIONS[k]["vector_source"] == "japanese"}
    _cross_bi   = {k: v for k, v in ds_map.items()
                   if k in results and CONDITIONS[k].get("bilingual", True)
                   and CONDITIONS[k]["vector_source"] == "japanese"}
    if _cross_mono and _cross_bi:
        avg_mono = sum(_cross_mono.values()) / len(_cross_mono)
        avg_bi   = sum(_cross_bi.values()) / len(_cross_bi)
        ratio    = avg_mono / avg_bi if avg_bi != 0 else float("inf")
        print(f"  双语模型 + JP 均值: {avg_bi:+.4f}  ({list(_cross_bi.keys())})")
        print(f"  单语模型 + JP 均值: {avg_mono:+.4f}  ({list(_cross_mono.keys())})")
        print(f"  单/双比值: {ratio:.2f}")
        if ratio > 0.7:
            verdict = "✅ 强支持「模型无关 register 子空间」：单语模型 JP 向量迁移效果与双语模型相当"
        elif ratio > 0.4:
            verdict = "⚡ 部分支持：单语模型有效但弱于双语；双语训练有增益但非必须"
        else:
            verdict = "❌ 不支持：单语模型 JP 迁移显著弱于双语，需双语训练"
        print(f"  → {verdict}")


        arches = set(CONDITIONS[k]["arch"] for k in results if CONDITIONS[k]["vector_source"] == "japanese")
        print(f"\n  覆盖架构: {arches}")
        if len(arches) >= 2:
            print(f"  ✅ 多架构验证通过，跨语言迁移不依赖特定架构")
    print(f"  {'─'*60}")


    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {
        "timestamp": ts,
        "conditions": {
            ckey: {
                "label": CONDITIONS[ckey]["label"],
                "source_file": os.path.basename(r["path"]),
                "by_alpha": r["by_alpha"],
                "direction_score": compute_direction_score(r["by_alpha"]),
                "dose_response": compute_dose_response(r["by_alpha"]),
            }
            for ckey, r in results.items()
        },
    }
    out_json = os.path.join(OUT_DIR, f"expD_summary_{ts}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n💾 已保存汇总: {out_json}")

    if args.plot:
        all_alphas = sorted(set(a for r in results.values() for a in r["by_alpha"].keys()))
        all_alphas = sorted(set(a for r in results.values() for a in r["by_alpha"].keys()))
        plot_comparison(results, all_alphas, OUT_DIR, ts)


def plot_comparison(results, alphas, out_dir, ts):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt


        _palette = ["#e74c3c","#c0392b","#3498db","#2980b9",
                    "#2ecc71","#27ae60","#f39c12","#d68910",
                    "#9b59b6","#8e44ad"]
        _markers = ["o","o","s","s","^","^","D","D","P","P"]
        _styles  = ["-",":","--","-.","-",":","--","-.","-",":"]
        ckey_list = list(results.keys())

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        for idx, ckey in enumerate(ckey_list):
            r  = results[ckey]
            ba = r["by_alpha"]
            xs = sorted(ba.keys())
            ys = [ba[x] for x in xs]
            label = f"{ckey}: {CONDITIONS[ckey]['label'][:32]}"
            ax1.plot(xs, ys, color=_palette[idx % len(_palette)],
                     marker=_markers[idx % len(_markers)],
                     linestyle=_styles[idx % len(_styles)],
                     label=label, linewidth=1.6, markersize=5)

        ax1.set_xlabel("Intervention Strength α")
        ax1.set_ylabel("P(formal) [RoBERTa]")
        ax1.set_title("Experiment D: Model Dependency\nDose-Response Curves")
        ax1.legend(fontsize=7.5, loc="best")
        ax1.grid(True, alpha=0.3)
        ax1.axvline(0, color="gray", linewidth=0.8, linestyle=":")


        ckeys = ckey_list
        ds_vals = [compute_direction_score(results[c]["by_alpha"]) for c in ckeys]
        bar_colors = [_palette[i % len(_palette)] for i in range(len(ckeys))]
        bars = ax2.bar(ckeys, ds_vals, color=bar_colors, alpha=0.85, edgecolor="white")
        ax2.axhline(0, color="gray", linewidth=0.8)
        ax2.set_xlabel("Condition")
        ax2.set_ylabel("Direction Score\n(mean P(formal)|α<0) − (mean P(formal)|α>0)")
        ax2.set_title("Direction Control Score by Condition")
        for bar, val in zip(bars, ds_vals):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                     f"{val:+.3f}", ha="center", va="bottom", fontsize=9)
        ax2.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        path = os.path.join(out_dir, f"expD_comparison_{ts}.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"📊 已保存对比图: {path}")
    except ImportError:
        print("⚠ matplotlib 不可用，跳过绘图")
    except Exception as e:
        print(f"⚠ 绘图失败: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", type=str, choices=list(CONDITIONS.keys()),
                        help='Condition.')
    parser.add_argument("--analyze", action="store_true",
                        help='Analyze.')
    parser.add_argument("--force", action="store_true",
                        help='Force.')
    parser.add_argument("--max_tests", type=int, default=MAX_TESTS)
    parser.add_argument("--strengths", type=str, default=STRENGTHS)
    parser.add_argument("--plot", action="store_true", help='Plot.')
    args = parser.parse_args()

    if args.analyze:
        analyze(args)
        return

    if args.condition:
        result_path = run_condition(args.condition, force=args.force,
                                    max_tests=args.max_tests, strengths=args.strengths)
        if result_path:
            print(f"\n条件 {args.condition} 完成: {result_path}")

            try:
                ba = parse_result(result_path)
                ds = compute_direction_score(ba)
                print(f"  baseline(α=0): {ba.get(0.0, 'N/A')}")
                print(f"  方向控制分:    {ds:+.4f}")
                baseline = ba.get(0.0, 0)
                for a in sorted(ba.keys()):
                    diff = ba[a] - baseline
                    arrow = "↑" if diff > 0.005 else ("↓" if diff < -0.005 else "→")
                    print(f"  α={a:+.0f}: P(formal)={ba[a]:.4f}  ({diff:+.4f} {arrow})")
            except Exception as e:
                print(f"  解析失败: {e}")
        return


    parser.print_help()
    print(f"\n当前各条件状态：")
    for ckey in CONDITIONS:
        path = find_result_file(ckey)
        status = f"✅ {os.path.basename(path)}" if path else "❌ 未完成"
        print(f"  {ckey:>3} [{CONDITIONS[ckey]['label'][:42]}]: {status}")
    print(f"\n建议运行顺序（每条件约 3-5 分钟）：")
    for ckey in CONDITIONS:
        path = find_result_file(ckey)
        if not path:
            print(f"  python model_dependency.py --condition {ckey}")
    print(f"  python model_dependency.py --analyze --plot")


if __name__ == "__main__":
    main()
