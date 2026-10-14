"""Check a real portable bundle and its local HTTP API, then save a public example.

The query is constructed from catalog products, not a person's review history.
This is a functional check; it does not estimate recommendation accuracy.
"""

from __future__ import annotations

import argparse
import json
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import torch

from beautyrec.serving import STRATEGIES, RecommendationEngine, make_server

ROOT = Path(__file__).resolve().parents[1]


def check_service(bundle: Path) -> dict:
    torch.set_num_threads(4)
    engine = RecommendationEngine(bundle, device="cpu")
    assert engine.manifest["serving_seed"] == 42
    assert not engine.manifest["diagnostic_only"]
    # Include both warm and new products, testing the two history projections.
    history = [engine.catalog[index]["asin"] for index in (0, 1, engine.warm_items)]
    server = make_server(engine, port=0)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    address = f"http://127.0.0.1:{server.server_address[1]}"
    responses = {}
    try:
        with urlopen(address + "/health", timeout=30) as response:
            health = json.load(response)
        assert health["status"] == "ok" and not health["diagnostic_only"]
        assert health["warm_items"] == engine.warm_items
        assert health["available_items"] == len(engine.catalog)
        for strategy in STRATEGIES:
            body = {"history": history, "top_k": 5, "strategy": strategy}
            request = Request(
                address + "/recommend",
                data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urlopen(request, timeout=120) as response:
                recommendation = json.load(response)
            items = recommendation["recommendations"]
            assert len(items) == 5
            assert [item["rank"] for item in items] == list(range(1, 6))
            assert len({item["asin"] for item in items}) == 5
            assert not set(history) & {item["asin"] for item in items}
            if strategy in ("generative", "sasrec"):
                assert not any(item["new_item"] for item in items)
            responses[strategy] = recommendation
            print(f"Verified CPU HTTP strategy: {strategy}", flush=True)
        malformed = Request(
            address + "/recommend",
            data=b'{"history": ["unknown-asin"]}',
            headers={"Content-Type": "application/json"},
        )
        try:
            urlopen(malformed, timeout=30)
        except HTTPError as error:
            assert error.code == 400
        else:
            raise AssertionError("Unknown ASIN should return HTTP 400.")
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=10)
    assert not worker.is_alive()
    return {
        "verified": True,
        "purpose": "Functional example from constructed catalog history; not accuracy evidence.",
        "protocol_fingerprint": engine.manifest["protocol_fingerprint"],
        "device": "cpu",
        "serving_seed": 42,
        "bundle_file_hashes": engine.manifest["files"],
        "health": health,
        "request": {"history": history, "top_k": 5},
        "responses": responses,
        "unknown_asin_http_status": 400,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT / "results/m5/bundle")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/examples/m5_inference.json")
    args = parser.parse_args()
    evidence = check_service(args.bundle)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", "utf-8")
    print(f"Verified real bundle and API; wrote {args.output}")


if __name__ == "__main__":
    main()
