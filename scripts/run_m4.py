"""Thin command-line entry point for the isolated M4 capacity experiment."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from beautyrec.m4_benchmark import run_m4  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/m4.json")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "results/m4")
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts/beauty")
    parser.add_argument("--device")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = run_m4(args.processed_dir, args.artifact_dir, args.run_dir, config, args.device)
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("m4_complete", "diagnostic_only", "common_vocab_size", "test_examples")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
