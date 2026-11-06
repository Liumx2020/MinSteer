"""Run a named historical experiment from its required working directory."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    entries = json.loads((ROOT / "configs/experiments.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", nargs="?", choices=sorted(entries))
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args, extra = parser.parse_known_args()
    if args.list or not args.experiment:
        for name, entry in entries.items():
            print(f"{name}: {entry['description']}")
        return
    entry = entries[args.experiment]
    command = [sys.executable, str(ROOT / entry["script"]), *entry.get("args", []), *extra]
    cwd = ROOT / entry.get("cwd", ".")
    if args.dry_run:
        print("cwd:", cwd)
        print(shlex.join(command))
        return
    if entry.get("gpu", True) and not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise SystemExit("Set CUDA_VISIBLE_DEVICES to an available GPU before running this experiment.")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    raise SystemExit(subprocess.run(command, cwd=cwd, env=env).returncode)


if __name__ == "__main__":
    main()
