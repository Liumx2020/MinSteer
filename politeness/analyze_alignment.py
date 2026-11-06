#!/usr/bin/env python
# -*- coding: utf-8 -*-


from release_paths import project_path
import os
import json
import argparse
import glob
import torch
from datetime import datetime

BASE_DIR = project_path('politeness')
JP_DIR = os.path.join(BASE_DIR, "extract_cached")
EN_DIR = os.path.join(BASE_DIR, "vectors_english")
OUT_DIR = os.path.join(BASE_DIR, "8_方向对齐")
os.makedirs(OUT_DIR, exist_ok=True)


def load_japanese_vector(pt_path, layer, residual_type="resid_pre"):
    data = torch.load(pt_path, map_location="cpu")
    r1 = data["generation_1"]["result"]["residuals"][residual_type]
    r2 = data["generation_2"]["result"]["residuals"][residual_type]
    g1 = r1[:, layer, :].float().mean(dim=0)
    g2 = r2[:, layer, :].float().mean(dim=0)
    return g2 - g1


def load_english_vector(pt_path, layer, residual_type="resid_pre"):
    data = torch.load(pt_path, map_location="cpu")
    vec = data["vectors"].get(residual_type)
    if vec is None:
        return None
    return vec[layer, :].float()


def cosine_similarity(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    return (a @ b) / (a.norm() * b.norm() + 1e-9)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jp_pt", type=str, default=None, help='Jp pt.')
    parser.add_argument("--en_pt", type=str, default=None, help='En pt.')
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument("--residual_type", type=str, default="resid_pre")
    args = parser.parse_args()


    if args.jp_pt is None:
        jp_files = glob.glob(os.path.join(JP_DIR, "swallow_argmax_detailed_*_従業員_to_上司.pt"))
        args.jp_pt = jp_files[0] if jp_files else None
    if args.en_pt is None:
        en_files = glob.glob(os.path.join(EN_DIR, "english_formal_casual_swallow_*.pt"))
        args.en_pt = en_files[0] if en_files else None

    if not args.jp_pt or not os.path.exists(args.jp_pt):
        print(f"❌ 日语向量不存在: {args.jp_pt}\n请先运行脚本 3")
        return
    if not args.en_pt or not os.path.exists(args.en_pt):
        print(f"❌ 英语向量不存在: {args.en_pt}\n请先运行: python3 extract_multilingual.py --model swallow")
        return

    jp_vec = load_japanese_vector(args.jp_pt, args.layer, args.residual_type)
    en_vec = load_english_vector(args.en_pt, args.layer, args.residual_type)
    if en_vec is None:
        print("❌ 英语向量格式不支持")
        return

    cos_sim = cosine_similarity(jp_vec, en_vec).item()
    result = {
        "japanese_vector": os.path.basename(args.jp_pt),
        "english_vector": os.path.basename(args.en_pt),
        "layer": args.layer,
        "residual_type": args.residual_type,
        "cosine_similarity": round(cos_sim, 6),
        "interpretation": "高相似度(>0.5)支持'共享方向'假设" if cos_sim > 0.5 else "低相似度，方向可能不同或需更多样本",
        "timestamp": datetime.now().isoformat(),
    }
    out_path = os.path.join(OUT_DIR, "direction_alignment.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("=" * 50)
    print("方向对齐：日语敬语 vs 英语 formal-casual")
    print("=" * 50)
    print(f"  日语: {result['japanese_vector']}")
    print(f"  英语: {result['english_vector']}")
    print(f"  层={args.layer}, {args.residual_type}")
    print(f"  余弦相似度: {cos_sim:.4f}")
    print(f"  → {result['interpretation']}")
    print(f"\n💾 已保存: {out_path}")


if __name__ == "__main__":
    main()
