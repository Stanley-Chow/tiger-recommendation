"""Recommend from an exported bundle or serve its loopback JSON API."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from beautyrec.serving import STRATEGIES, RecommendationEngine, make_server  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT / "results/m5/bundle")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--history", nargs="+")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--strategy", choices=STRATEGIES, default="generative_content")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    engine = RecommendationEngine(args.bundle, args.device)
    if args.serve:
        server = make_server(engine, port=args.port)
        print(f"Serving http://127.0.0.1:{args.port}; Ctrl-C to stop.", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
    elif args.history:
        print(
            json.dumps(
                engine.recommend(args.history, args.top_k, args.strategy),
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        parser.error("Provide --history ASIN ... or --serve.")


if __name__ == "__main__":
    main()
