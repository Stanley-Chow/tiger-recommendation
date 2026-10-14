"""Run the complete synthetic pipeline without a dataset download or GPU."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from beautyrec.m5_demo import run_demo  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "results/m5_demo")
    args = parser.parse_args()
    result = run_demo(args.directory, ROOT / "configs/m5.json")
    print(
        json.dumps(
            {key: result[key] for key in ("m5_complete", "diagnostic_only", "serving_bundle")},
            indent=2,
        )
    )
