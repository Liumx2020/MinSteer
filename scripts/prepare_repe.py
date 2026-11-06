"""Apply the recorded one-line RepE tokenizer compatibility patch after checkout."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def main():
    record = json.loads((ROOT / "configs/external_sources.json").read_text())["repe"]
    path = ROOT / record["file"]
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest == record["patched_sha256"]:
        print("Recorded RepE compatibility patch already present.")
        return
    if digest != record["upstream_sha256"]:
        raise RuntimeError("Unexpected RepE source. Use the frozen commit; refusing to patch a different file.")
    data = data.replace(b'return_tensors=self.framework', b'return_tensors="pt"')
    assert hashlib.sha256(data).hexdigest() == record["patched_sha256"]
    path.write_bytes(data)
    print("Applied recorded tokenizer tensor-backend patch; PCA estimator unchanged.")


if __name__ == "__main__":
    main()
