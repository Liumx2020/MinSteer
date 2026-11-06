"""Check frozen prompts, input contracts, entry points and CPU steering arithmetic."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import io
import tokenize
import importlib.util
import math

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "experiments/sentiment_scaling"


def definitions(path, names, namespace):
    tree = ast.parse(path.read_text())
    selected = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert len(selected) == len(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-torch", action="store_true")
    args = parser.parse_args()
    files = [ROOT / "release_paths.py"]
    for directory in ["analysis", "baselines", "experiments", "politeness", "sentiment", "toxicity", "scripts", "tests"]:
        files += [p for p in (ROOT / directory).rglob("*.py")
                  if not set(p.parts) & {"representation-engineering", "official_repo", "__pycache__"}]
    for path in files:
        compile(path.read_text(), str(path), "exec")
        assert path.relative_to(ROOT).as_posix().isascii(), path
        source = path.read_text()
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type == tokenize.COMMENT:
                assert not re.search(r'[\u3400-\u9fff]', token.string), (path, token.start)
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                doc = ast.get_docstring(node, clean=False)
                assert doc is None or not re.search(r'[\u3400-\u9fff]', doc), path
    rows = json.loads((ROOT / "prompts/sentiment_pool50.json").read_text())
    assert len(rows) == 50
    assert len({(r["system"], r["user"], r["conversion"]) for r in rows}) == 50
    ns = definitions(ROOT / "sentiment/extract_cached.py", {"create_emotion_prompt", "create_emotion_conversion_prompt"}, {})
    for row in rows:
        system, user = ns["create_emotion_prompt"]("negative", row["scenario"])
        assert row["system"] == system
        if row["sample_idx"] == 0:
            assert row["user"] == user
        else:
            assert row["user"].startswith(user + "\nContext: ")
        assert row["conversion"] == ns["create_emotion_conversion_prompt"]("positive", row["scenario"])
    contract = json.loads((ROOT / "configs/sentiment_scaling.json").read_text())
    assert hashlib.sha256((ROOT / "prompts/sentiment_pool50.json").read_bytes()).hexdigest() == contract["prompt_pool_sha256"]
    assert hashlib.sha256((ROOT / "sentiment/extract_cached.py").read_bytes()).hexdigest() == contract["extraction_source_sha256"]
    data = json.loads((ROOT / "configs/data_contract.json").read_text())
    assert len(data["cached_eval500_row_sha256"]) == 500
    assert len(data["stanford99"]) == 99
    assert len(data["paradetox"]["train_pairs_candidates"]) == 1000
    assert len(data["paradetox"]["eval_samples"]) == 100
    path = EXT / 'subsets.py'
    spec = importlib.util.spec_from_file_location('n5_subsets', path)
    subsets = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(subsets)
    manifest = subsets.checked_manifest()
    expected_indices = [[1, 0, 4, 3, 3], [0, 4, 2, 7, 5], [6, 8, 8, 1, 2],
                        [4, 6, 7, 4, 1], [1, 6, 0, 9, 9]]
    assert [[x['sample_idx'] for x in group['choices']] for group in manifest['groups']] == expected_indices
    assert all(len({x['scenario'] for x in group['choices']}) == 5 for group in manifest['groups'])
    reference = json.loads((ROOT / 'results/sentiment_n5_subsets.json').read_text())
    summary = subsets.summarize_groups(reference['groups'])
    assert summary == reference
    assert math.isclose(summary['mean_positive_rate'], 0.7744, abs_tol=1e-15)
    assert math.isclose(summary['sample_sd_positive_rate'], 0.027291024165465083, abs_tol=1e-15)
    try:
        subsets.summarize_groups(reference['groups'][:-1])
    except ValueError:
        pass
    else:
        raise AssertionError('An incomplete set of subset results was accepted')
    path = ROOT / 'experiments/extraction_variance/summarize.py'
    spec = importlib.util.spec_from_file_location('multiseed_summary', path)
    multiseed = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(multiseed)
    reference = json.loads((ROOT / 'results/sentiment_multiseed.json').read_text())
    records = {method: reference['rows'][method]['by_seed'] for method in multiseed.METHOD_N}
    assert multiseed.summarize(records) == reference
    assert reference['rows']['repe_n1']['status'] == 'N/A'
    assert reference['rows']['caa_class_n50']['mean_percent'] == 91.2
    assert math.isclose(reference['rows']['caa_class_n50']['sample_sd_percentage_points'], 3.605551275463989)
    controls = json.loads((ROOT / 'results/control_experiments.json').read_text())
    assert controls['tab:sentiment_seed_variance']['reference'] == 'results/sentiment_multiseed.json'
    for entry in json.loads((ROOT / "configs/experiments.json").read_text()).values():
        assert (ROOT / entry["script"]).is_file()
        assert (ROOT / entry.get("cwd", ".")).is_dir()
    forbidden = re.compile(r"/(?:home|data[0-9]?)/[^\s'\"]+|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]+|sk-[A-Za-z0-9]{24,}")
    public = files + [ROOT / "README.md", ROOT / "THIRD_PARTY.md", ROOT / "requirements.txt"]
    for directory, pattern in [("configs", "*.json"), ("results", "*.json"), ("prompts", "*.json"), ("docs", "*.md")]:
        public += list((ROOT / directory).glob(pattern))
    public += [ROOT / "politeness/chinese_requests.json", ROOT / "prompts/politeness_scenarios.json"]
    for path in public:
        assert path.relative_to(ROOT).as_posix().isascii(), path
        if path.suffix in {".py", ".json", ".md", ".txt"}:
            assert not forbidden.search(path.read_text()), path
        if path.suffix == '.md':
            for line in path.read_text().splitlines():
                if re.match(r'^#{1,6} ', line):
                    assert not re.search(r'[\u3400-\u9fff]', line), path
    if args.with_torch:
        import torch
        import glob
        import os
        import tempfile
        from types import SimpleNamespace
        class Tokens(dict):
            def to(self, _device):
                return self
        class Tokenizer:
            padding_side = 'left'
            def __call__(self, *_args, **_kwargs):
                return Tokens(attention_mask=torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]]))
        hidden = torch.arange(8).reshape(2, 4, 1)
        class Model:
            device = 'cpu'
            def __call__(self, **_kwargs):
                return SimpleNamespace(hidden_states=[None, hidden])
        ns = definitions(ROOT / 'experiments/extraction_variance/extract.py', {'last_prompt_activations'},
                         {'torch': torch, 'apply_chat': lambda *_: '', 'sentiment_messages': lambda *_: []})
        activations = ns['last_prompt_activations'](Model(), Tokenizer(), ['a', 'b'], 'positive', 0, 2)
        assert activations.tolist() == [[3.0], [7.0]]
        ns = definitions(EXT / "evaluate.py", {"SteeringHook"}, {"torch": torch})
        hook = ns["SteeringHook"](torch.tensor([1., -2.]), 1.5, "cpu")
        for length in [7, 1]:
            x = torch.zeros(1, length, 2, dtype=torch.bfloat16)
            y = hook(None, None, x)
            assert torch.equal(y, torch.tensor([1.5, -3.], dtype=x.dtype).expand_as(x))
            y_tuple = hook(None, None, (x, "cache"))
            assert torch.equal(y_tuple[0], y) and y_tuple[1] == "cache"

        a = torch.arange(24, dtype=torch.bfloat16).reshape(3, 2, 4)
        b = torch.arange(40, dtype=torch.bfloat16).reshape(5, 2, 4)
        assert (b.mean(0) - a.mean(0)).dtype == torch.bfloat16
        assert sorted(["student_0", "chef_1", "chef_0"])[0] == "chef_0"
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary)
            value = {"generation_1": {"result": {"residuals": {"resid_post": a, "resid_pre": a}}},
                     "generation_2": {"result": {"residuals": {"resid_post": b, "resid_pre": b}}}}
            first = raw / "emotion_style_llama3_en_chef_negative_argmax_sample_0.pt"
            second = raw / "emotion_style_llama3_en_office_worker_negative_argmax_sample_0.pt"
            torch.save(value, first)
            value2 = {"generation_1": value["generation_1"],
                      "generation_2": {"result": {"residuals": {"resid_post": b + 4}}}}
            torch.save(value2, second)
            ns = definitions(EXT / "average.py", {"average_vectors"},
                             {"torch": torch, "glob": glob, "os": os, "project_path": lambda _: str(raw)})
            assert torch.equal(ns["average_vectors"](1), b.mean(0) - a.mean(0))
            expected = torch.stack([b.mean(0) - a.mean(0), (b + 4).mean(0) - a.mean(0)]).mean(0)
            assert torch.equal(ns["average_vectors"](2), expected)
            ns = definitions(ROOT / "politeness/evaluate_transfer.py", {"load_feature_vector"},
                             {"torch": torch, "HIDDEN_SIZE": 4})
            expected = b[:3, 1, :].mean(0) - a[:, 1, :].mean(0)
            assert torch.equal(ns["load_feature_vector"](str(first), 1, "resid_pre"), expected)
            full = ns['load_feature_vector'](str(first), 1, 'resid_pre', readout='full_mean')
            assert torch.equal(full, b.float().mean(0)[1] - a.float().mean(0)[1])
            assert not torch.equal(full, expected.float())

        from politeness.geometry import summarize_vectors
        jp = [torch.tensor(value) for value in [[1., 0.], [-1., 0.], [1., 1.], [-1., 1.], [0., 1.]]]
        geometry = summarize_vectors(jp, torch.tensor([1., 0.]), torch.tensor([0., 1.]), 1)
        assert geometry['five_jp_comparison']['jp_count'] == 5
        assert abs(geometry['five_jp_comparison']['last_mean']) < 1e-7
        assert geometry['single_jp_reference']['last_cosine'] == -1.0
        try:
            summarize_vectors(jp[:4], jp[0], jp[1])
        except ValueError:
            pass
        else:
            raise AssertionError('A five-vector statistic accepted only four JP vectors')

        difference = torch.tensor([[2., 3., 4.]])
        assert torch.count_nonzero(difference - difference.mean(0)) == 0
    print(json.dumps({"python_files_compiled": len(files), "distinct_prompts": 50, "tensor_checks_cpu": args.with_torch, "gpu_used": False}))


if __name__ == "__main__":
    main()
