"""Validate the pinned two-/three-number curriculum and four-number OOD splits."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from src.prompts import PREFILL, TEMPLATES
from src.reward.verify import check_equation


def validate_curriculum(data_dir):
    """Check the benchmark on CPU and return its manifest."""
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    protocol = manifest["protocol"]
    method = protocol["prompt_method"]
    if TEMPLATES[method] != protocol["prompt_template"] or PREFILL[method] != protocol["assistant_prefill"]:
        raise ValueError("Curriculum prompt or prefill differs from the benchmark manifest.")

    seen = {}
    for split in ("train", "val", "test", "test_ood"):
        expected = manifest["splits"][split]
        path = data_dir / expected["file"]
        raw = path.read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(raw).hexdigest() != expected["sha256"]:
            raise ValueError(f"Dataset hash differs from the benchmark manifest: {path}")

        counts = Counter()
        ids = set()
        keys = set()
        for line in raw.splitlines():
            row = json.loads(line)
            numbers, target = row["numbers"], row["target"]
            key = (tuple(sorted(numbers)), target)
            if key in keys or row["id"] in ids:
                raise ValueError(f"Duplicate puzzle or ID in {split}: {row['id']}")
            if key in seen:
                raise ValueError(f"Puzzle overlaps between {seen[key]} and {split}: {row['id']}")
            if not check_equation(row["solution"], numbers, target):
                raise ValueError(f"Invalid reference solution in {split}: {row['id']}")
            ids.add(row["id"])
            keys.add(key)
            counts[str(len(numbers))] += 1

        levels = {"4"} if split == "test_ood" else {"2", "3"}
        if set(counts) != levels or counts != expected["by_input_count"] or len(keys) != expected["puzzles"]:
            raise ValueError(f"Puzzle counts differ from the benchmark manifest: {split}")
        seen.update(dict.fromkeys(keys, split))

    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/curriculum"))
    args = parser.parse_args()
    print(json.dumps(validate_curriculum(args.data_dir), indent=2))


if __name__ == "__main__":
    main()

