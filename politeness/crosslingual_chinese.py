#!/usr/bin/env python
# -*- coding: utf-8 -*-
from release_paths import project_path

import os
import sys
import json
import glob
import argparse
import subprocess
import datetime

BASE_DIR = project_path('politeness')
SCRIPT4 = os.path.join(BASE_DIR, "evaluate_transfer.py")
SCRIPT7 = os.path.join(BASE_DIR, "extract_multilingual.py")
OUT_DIR = os.path.join(BASE_DIR, "crosslingual_results")
CN_VEC_DIR = os.path.join(BASE_DIR, "vectors_chinese")
os.makedirs(OUT_DIR, exist_ok=True)

STRENGTHS = "0,1,-1,2,-2"
MAX_CN = 30
MAX_EN = 33

EXPERIMENTS = {
    "E1": {
        "label": "JP向量 + LLaMA3-ZH + 中文prompt → 中文输出",
        "model": "llama3_zh",
        "corpus": "cn",
        "vector_source": "japanese",
        "max_tests": MAX_CN,
        "stratify": "1,0,-1",
    },
    "E2": {
        "label": "CN向量 + LLaMA3-ZH + 中文prompt → 中文输出",
        "model": "llama3_zh",
        "corpus": "cn",
        "vector_source": "chinese",
        "max_tests": MAX_CN,
        "stratify": "1,0,-1",
    },
    "E3": {
        "label": "JP向量 + LLaMA3-ZH + 英文prompt → 英文输出",
        "model": "llama3_zh",
        "corpus": "stanford",
        "vector_source": "japanese",
        "max_tests": MAX_EN,
        "stratify": "1,0,-1",
    },
    "E4": {
        "label": "CN向量 + LLaMA3-ZH + 英文prompt → 英文输出",
        "model": "llama3_zh",
        "corpus": "stanford",
        "vector_source": "chinese",
        "max_tests": MAX_EN,
        "stratify": "1,0,-1",
    },
}


def ensure_chinese_vector():
    path = os.path.join(CN_VEC_DIR, "chinese_formal_casual_llama3_zh_layer15.pt")
    if os.path.exists(path):
        return path
    print("⚠ 中文向量不存在，正在提取（需加载 LLaMA3-ZH）...")
    ret = subprocess.run(
        [sys.executable, SCRIPT7, "--language", "cn", "--model", "llama3_zh", "--max_prompts", "5"],
        cwd=BASE_DIR,
    )
    return path if ret.returncode == 0 and os.path.exists(path) else None


def run_experiment(exp_id):
    cfg = EXPERIMENTS[exp_id]
    if cfg["vector_source"] == "chinese":
        if ensure_chinese_vector() is None:
            print(f"❌ E2/E4 需要中文向量，提取失败，跳过 {exp_id}")
            return None
    cmd = [
        sys.executable, SCRIPT4,
        "--model", cfg["model"],
        "--corpus", cfg["corpus"],
        "--vector_source", cfg["vector_source"],
        "--max_tests", str(cfg["max_tests"]),
        "--strengths", STRENGTHS,
        "--prompt_mode", "neutral",
    ]
    if cfg.get("stratify"):
        cmd += ["--stratify", cfg["stratify"]]
    print(f"\n{'='*60}\n  运行 {exp_id}: {cfg['label']}\n{'='*60}")
    print("  执行:", " ".join(cmd))
    ret = subprocess.run(cmd, cwd=BASE_DIR)
    if ret.returncode != 0:
        print(f"  ❌ {exp_id} 运行失败")
        return None

    res_dir = os.path.join(BASE_DIR, "evaluate_transfer")
    vs = cfg["vector_source"]
    if vs == "chinese":
        pattern = f"cross_lang_llama3_zh_{cfg['corpus']}_cn_*.json"
    elif vs == "japanese":
        pattern = f"cross_lang_llama3_zh_{cfg['corpus']}_v*.json"
    else:
        pattern = f"cross_lang_llama3_zh_{cfg['corpus']}_en_*.json"
    files = sorted(glob.glob(os.path.join(res_dir, pattern)))
    if files:
        import shutil
        dest = os.path.join(OUT_DIR, f"{exp_id}_{os.path.basename(files[-1])}")
        shutil.copy2(files[-1], dest)
        print(f"  ✅ 结果已复制: {dest}")
        return dest
    print(f"  ⚠ 未找到输出文件")
    return None


def analyze():
    summary = {}
    for exp_id, cfg in EXPERIMENTS.items():

        pat1 = os.path.join(OUT_DIR, f"{exp_id}_*.json")
        vs = cfg["vector_source"]
        if vs == "chinese":
            pat2 = os.path.join(BASE_DIR, "evaluate_transfer", f"cross_lang_llama3_zh_{cfg['corpus']}_cn_*.json")
        elif vs == "japanese":
            pat2 = os.path.join(BASE_DIR, "evaluate_transfer", f"cross_lang_llama3_zh_{cfg['corpus']}_v*.json")
        else:
            pat2 = os.path.join(BASE_DIR, "evaluate_transfer", f"cross_lang_llama3_zh_{cfg['corpus']}_en_*.json")
        for pat in [pat1, pat2]:
            files = sorted(glob.glob(pat))
            if files:
                path = files[-1]
                try:
                    with open(path, encoding="utf-8") as f:
                        data = json.load(f)
                    fm = data.get("formality", {})
                    mean_by = fm.get("mean_by_alpha", {})
                    if mean_by:
                        summary[exp_id] = {
                            "label": cfg["label"],
                            "file": os.path.basename(path),
                            "mean_by_alpha": mean_by,
                            "evaluator": fm.get("evaluator", ""),
                        }
                except Exception as e:
                    print(f"  {exp_id}: 解析失败 {e}")
                break
    if not summary:
        print("未找到任何 E1–E4 结果文件")
        return
    print("\n" + "=" * 70)
    print("  E1–E4 汇总")
    print("=" * 70)
    for exp_id in ["E1", "E2", "E3", "E4"]:
        if exp_id not in summary:
            print(f"  {exp_id}: —")
            continue
        s = summary[exp_id]
        print(f"\n  {exp_id} {s['label']}")
        print(f"    评估: {s.get('evaluator', '')}")
        for a in [0.0, -1.0, -2.0, 1.0, 2.0]:
            v = s["mean_by_alpha"].get(a) or s["mean_by_alpha"].get(str(a))
            if v is not None:
                print(f"    α={a:+.1f}: P(formal)={v:.4f}")

    print("\n  方向控制分 (负α均值 − 正α均值):")
    for exp_id in ["E1", "E2", "E3", "E4"]:
        if exp_id not in summary:
            continue
        m = summary[exp_id]["mean_by_alpha"]
        def _v(k): return m.get(k) if k in m else m.get(str(k)) or m.get(str(float(k)))
        neg = [_v(-2), _v(-1)]
        pos = [_v(1), _v(2)]
        neg = [x for x in neg if x is not None]
        pos = [x for x in pos if x is not None]
        ds = (sum(neg) / len(neg) - sum(pos) / len(pos)) if neg and pos else 0.0
        print(f"    {exp_id}: {ds:+.4f}")
    out_path = os.path.join(OUT_DIR, f"E1E4_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n💾 汇总已保存: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default=None, help="E1 | E2 | E3 | E4 | E1,E2 | all")
    parser.add_argument("--analyze", action="store_true", help='Analyze.')
    args = parser.parse_args()
    if args.analyze:
        analyze()
        return
    if not args.run:
        parser.print_help()
        print("\n示例: --run E1  或  --run all  或  --analyze")
        return
    if args.run.lower() == "all":
        to_run = list(EXPERIMENTS.keys())
    else:
        to_run = [x.strip() for x in args.run.split(",") if x.strip() in EXPERIMENTS]
    if not to_run:
        print("无效的 --run，可选: E1, E2, E3, E4 或 all")
        return
    for exp_id in to_run:
        run_experiment(exp_id)


if __name__ == "__main__":
    main()
