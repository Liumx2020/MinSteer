"""Export the original truncated-mean keigo contrast for the matched unit-norm comparison."""
import argparse
import importlib.util
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--layer", type=int, default=15)
    args = parser.parse_args()
    path = ROOT / "politeness/evaluate_transfer.py"
    spec = importlib.util.spec_from_file_location("original_keigo", path)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    vector = original.load_feature_vector(str(args.input), args.layer, "resid_pre")
    if vector is None:
        raise FileNotFoundError(args.input)
    # The matched follow-up uses unit norm; the legacy evaluator keeps its raw scale.
    vector = vector.float()
    vector = vector / vector.norm().clamp_min(1e-8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(vector, args.output)


if __name__ == "__main__":
    main()
