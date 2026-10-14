"""Thin entry point for M5 feature construction and the frozen comparison."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from beautyrec.m5_benchmark import run_m5  # noqa: E402
from beautyrec.m5_features import build_m5_features  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/m5.json")
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data/amazon2023/processed")
    parser.add_argument("--feature-dir", type=Path, default=ROOT / "artifacts/amazon2023")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "results/m5")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    from beautyrec.training import resolve_device

    device = str(resolve_device(args.device))
    config = json.loads(args.config.read_text("utf-8"))
    build_m5_features(
        args.processed_dir,
        args.feature_dir,
        model_path=args.model_path,
        device=device,
        offline=args.offline,
        **{k: v for k, v in config["features"].items() if k != "model_name"},
    )
    result = run_m5(args.processed_dir, args.feature_dir, args.run_dir, config, device)
    print(
        json.dumps(
            {k: result[k] for k in ("m5_complete", "diagnostic_only", "serving_bundle")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
