#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'Adaptive toxicity intervention using layer-wise logistic concept probes (GCAV reference).'
import os
import sys
import json
import argparse

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_TOX_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), "toxicity")
sys.path.insert(0, _TOX_DIR)

import run_all_toxicity_eval as ev  # noqa: E402

ev._lazy_imports()
torch = ev.torch

NEUTRAL_PROMPT_USER = "Please write a short message:"


def collect_states(model, tokenizer, path, texts, layer_ids):
    msgs = [{"role": "user", "content": NEUTRAL_PROMPT_USER}]
    prefix = ev._apply_chat_template(tokenizer, msgs, path=path)
    layers = ev.get_layers(model)
    out = {l: [] for l in layer_ids}
    for t in ev.tqdm(texts, desc="collect states"):
        ids = tokenizer(prefix + t, return_tensors="pt",
                        add_special_tokens=False).input_ids.to(model.device)
        captured = {}

        def hf(li):
            def hook(module, inputs, output):
                h = output[0] if isinstance(output, tuple) else output
                captured[li] = h[:, -1, :].detach().float().cpu()
            return hook

        hooks = [layers[l].register_forward_hook(hf(l)) for l in layer_ids]
        with torch.no_grad():
            model(ids, use_cache=False)
        for h in hooks:
            h.remove()
        for l in layer_ids:
            out[l].append(captured[l].squeeze(0))
    return {l: torch.stack(v) for l, v in out.items()}


def train_probe(X_pos, X_neg, epochs=200, lr=0.05, wd=1e-3):
    X = torch.cat([X_pos, X_neg]).float()
    y = torch.cat([torch.ones(len(X_pos)), torch.zeros(len(X_neg))])
    mu, sd = X.mean(0, keepdim=True), X.std(0, keepdim=True).clamp(min=1e-6)
    Xn = (X - mu) / sd
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr, weight_decay=wd)
    for _ in range(epochs):
        opt.zero_grad()
        logits = Xn @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = (((Xn @ w + b) > 0).float() == y).float().mean().item()

        w_orig = (w / sd.squeeze(0)).detach()
        b_orig = (b - (w * (mu.squeeze(0) / sd.squeeze(0))).sum()).detach()
    return w_orig, b_orig, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alias", type=str, default="Llama-3.1-8B")
    ap.add_argument("--m_train", type=int, default=100, help='M train.')
    ap.add_argument("--target_p", type=float, default=0.01, help='Target p.')
    ap.add_argument("--layers", type=str, default="mid",
                    help='Layers.')
    ap.add_argument("--max_eps_norm", type=float, default=8.0, help='Max eps norm.')
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = next((c for c in ev.MODELS_CONFIG if c["alias"] == args.alias), None)
    if cfg is None:
        sys.exit(f"未知 alias: {args.alias}")
    path = cfg["path"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = ev.AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = ev.AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=(torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16), device_map={"": device}, trust_remote_code=True)
    model.eval()
    n_layers = len(ev.get_layers(model))

    if args.layers == "mid":
        layer_ids = list(range(n_layers // 3, 2 * n_layers // 3))
    elif args.layers == "all":
        layer_ids = list(range(n_layers))
    else:
        layer_ids = [int(x) for x in args.layers.split(",")]


    cache_path = os.path.join(
        ev.OUTPUT_DIR,
        f"paradetox_splits_seed{ev.PD_SEED}_train{ev.N_PD_TRAIN_CANDS}_eval{ev.N_PD_EVAL}.json")
    train_pairs, eval_samples = ev.load_paradetox_splits(
        n_train_cands=ev.N_PD_TRAIN_CANDS, n_eval=ev.N_PD_EVAL, seed=ev.PD_SEED,
        cache_path=cache_path if os.path.exists(cache_path) else None)
    import random as _rd
    _rd.Random(args.seed).shuffle(train_pairs)
    pairs = train_pairs[:args.m_train]


    tox_states = collect_states(model, tokenizer, path, [p[0] for p in pairs], layer_ids)
    neu_states = collect_states(model, tokenizer, path, [p[1] for p in pairs], layer_ids)
    probes = {}
    for l in layer_ids:
        w, b, acc = train_probe(tox_states[l], neu_states[l])
        probes[l] = (w, b, acc)
    accs = [probes[l][2] for l in layer_ids]
    print(f"概念分类器 acc：mean={sum(accs)/len(accs):.3f} "
          f"min={min(accs):.3f} max={max(accs):.3f}（layers {layer_ids[0]}..{layer_ids[-1]}）")


    logit_target = torch.logit(torch.tensor(args.target_p)).item()
    layers = ev.get_layers(model)

    def make_hook(l):
        w, b, _ = probes[l]
        w_d = w.to(device=device, dtype=torch.float32)
        b_d = b.to(device=device, dtype=torch.float32)
        w_sq = (w_d @ w_d).clamp(min=1e-8)

        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            hv = h[:, -1, :].float()
            cur = hv @ w_d + b_d
            need = (logit_target - cur) / w_sq
            eps = need.unsqueeze(-1) * w_d.unsqueeze(0)
            nrm = eps.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            scale = torch.clamp(nrm, max=args.max_eps_norm) / nrm
            eps = eps * scale
            h = h.clone()
            h[:, -1, :] += eps.to(h.dtype)
            return (h,) + out[1:] if isinstance(out, tuple) else h
        return hook


    det = ev.Detoxify('unbiased', device=str(device)) if ev.Detoxify else None
    results = []
    handles = [layers[l].register_forward_hook(make_hook(l)) for l in layer_ids]
    try:
        for ex in ev.tqdm(eval_samples, desc="GCAV eval"):
            fmt = ev._format_rewrite_prompt(tokenizer, ex["toxic"], path)
            ids = tokenizer(fmt, return_tensors="pt").to(device).input_ids
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=ev.MAX_NEW_TOKENS, do_sample=False,
                                     pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
            text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
            results.append({
                "toxic_text": ex["toxic"],
                "neutral_reference": ex.get("neutral_reference"),
                "baseline_or_intervention_output": text,
            })
    finally:
        for h in handles:
            h.remove()

    for r in results:
        r["detoxify_toxicity"] = (float(det.predict(r["baseline_or_intervention_output"])["toxicity"])
                                  if det and r["baseline_or_intervention_output"].strip() else None)
    avg = ev._avg(results)
    out_path = os.path.join(ev.OUTPUT_DIR,
                            f"{args.alias}_paradetox_pos_gcav_p{args.target_p}.json")
    with open(out_path, "w") as f:
        json.dump({
            "model": args.alias, "path": path,
            "label": f"paradetox_pos_gcav_p{args.target_p}",
            "method": "GCAV (Zhang et al., AAAI 2025)",
            "layers": layer_ids, "target_p": args.target_p,
            "m_train": args.m_train, "probe_acc_mean": sum(accs) / len(accs),
            "eval_dataset": "ParaDetox_rewrite", "eval_metric": "Detoxify toxicity",
            "avg_detoxify_toxicity": avg, "n_prompts": len(results),
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"GCAV avg Detoxify = {avg:.5f}  ->  {out_path}")
    print("STA/SIM/FL: python toxicity/eval_paradetox_metrics.py 会自动扫描该文件")


if __name__ == "__main__":
    main()
