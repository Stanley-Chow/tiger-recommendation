"""Download or prepare the isolated Amazon 2023 M5 data stage."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from beautyrec.amazon2023 import download_amazon2023  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["download", "prepare"])
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/amazon2023/raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/amazon2023/processed")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/m5.json")
    args = parser.parse_args()
    if args.command == "download":
        print(
            json.dumps({k: str(v) for k, v in download_amazon2023(args.raw_dir).items()}, indent=2)
        )
    else:
        from beautyrec.amazon2023 import TemporalCohortConfig, prepare_temporal_cohort

        config = json.loads(args.config.read_text(encoding="utf-8"))
        prepare_temporal_cohort(
            args.raw_dir, args.output_dir, TemporalCohortConfig(**config["data"])
        )
