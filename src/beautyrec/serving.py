"""Portable inference bundle, ASIN-based recommendations, and a loopback HTTP API."""

from __future__ import annotations

import json
import shutil
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch

from beautyrec.benchmark import _atomic_json, _file_hash
from beautyrec.content_retrieval import ContentRetriever, reciprocal_rank_fusion
from beautyrec.experiment_training import load_experiment_checkpoint
from beautyrec.retrieval import retrieve_rankings
from beautyrec.training import resolve_device

STRATEGIES = ("generative", "sasrec", "content", "generative_content", "sasrec_content")


def export_bundle(processed, features, run, selected, config, protocol_fingerprint):
    """Export the prespecified first seed, independent of test accuracy.

    Optimizer state is omitted. Source checkpoint hashes preserve the link back
    to the frozen experiment. Relative file names make the bundle relocatable.
    """
    bundle = Path(run) / "bundle"
    marker = bundle / "bundle.json"
    if marker.exists():
        saved = json.loads(marker.read_text("utf-8"))
        if saved["protocol_fingerprint"] != protocol_fingerprint:
            raise ValueError("Inference bundle belongs to a different experiment.")
        verify_bundle(bundle)
        return bundle
    bundle.mkdir(parents=True, exist_ok=True)
    for name in ("catalog.json", "available_catalog.json"):
        shutil.copy2(Path(processed) / name, bundle / name)
    shutil.copy2(Path(features) / "available_embeddings.npy", bundle / "available_embeddings.npy")
    shutil.copy2(
        Path(selected["generative"]["artifact_dir"]) / "semantic_ids.json",
        bundle / "semantic_ids.json",
    )
    for method in ("generative", "sasrec"):
        source = Path(selected[method]["checkpoint"])
        if _file_hash(source) != selected[method]["sha256"]:
            raise ValueError("Selected checkpoint changed before bundle export.")
        checkpoint = torch.load(source, map_location="cpu", weights_only=True)
        checkpoint.pop("optimizer_state", None)
        torch.save(checkpoint, bundle / f"{method}.pt")
    _atomic_json(
        marker,
        {
            "schema_version": 1,
            "protocol_fingerprint": protocol_fingerprint,
            "serving_seed": config["seeds"][0],
            "test_selected_seed": False,
            "max_history": config["training"]["max_history"],
            "beam_size": config["evaluation"]["beam_size"],
            "fusion": config["fusion"],
            "diagnostic_only": config["diagnostic_only"],
            "source_checkpoints": {method: selected[method]["sha256"] for method in selected},
            "files": {p.name: _file_hash(p) for p in sorted(bundle.iterdir()) if p.is_file()},
        },
    )
    return bundle


def verify_bundle(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "bundle.json").read_text("utf-8"))
    required = {
        "catalog.json",
        "available_catalog.json",
        "available_embeddings.npy",
        "semantic_ids.json",
        "generative.pt",
        "sasrec.pt",
    }
    if manifest.get("schema_version") != 1 or set(manifest["files"]) != required:
        raise ValueError("Unsupported or incomplete inference bundle.")
    for name, sha in manifest["files"].items():
        if _file_hash(directory / name) != sha:
            raise ValueError(f"Inference bundle checksum mismatch: {name}")
    return manifest


class RecommendationEngine:
    """Use a verified bundle without raw reviews, training data, or text downloads."""

    def __init__(self, bundle: Path, device="cpu"):
        self.bundle = Path(bundle)
        self.manifest = verify_bundle(self.bundle)
        self.device = resolve_device(device)
        self.catalog = json.loads((self.bundle / "available_catalog.json").read_text("utf-8"))
        warm = json.loads((self.bundle / "catalog.json").read_text("utf-8"))
        if self.catalog[: len(warm)] != warm or any(
            row["item_id"] != index for index, row in enumerate(self.catalog)
        ):
            raise ValueError("Inference catalog ordering is inconsistent.")
        self.warm_items = len(warm)
        self.asin_to_id = {row["asin"]: row["item_id"] for row in self.catalog}
        if len(self.asin_to_id) != len(self.catalog):
            raise ValueError("Inference catalog contains repeated ASINs.")
        vectors = np.load(self.bundle / "available_embeddings.npy", allow_pickle=False)
        if len(vectors) != len(self.catalog):
            raise ValueError("Inference features and catalog differ.")
        self.content = ContentRetriever(vectors, self.device)
        self.models = {}
        for method in ("generative", "sasrec"):
            model, book, _ = load_experiment_checkpoint(
                self.bundle / f"{method}.pt", self.bundle, self.device
            )
            self.models[method] = model
            self.codebook = book
        self.lock = threading.Lock()

    def recommend(self, history: list[str], top_k=20, strategy="generative_content") -> dict:
        if (
            not isinstance(history, list)
            or not 1 <= len(history) <= 10000
            or any(not isinstance(asin, str) or not asin for asin in history)
            or len(history) != len(set(history))
        ):
            raise ValueError("history must contain 1–10000 unique ASIN strings in time order.")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 200:
            raise ValueError("top_k must be an integer from 1 to 200.")
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {', '.join(STRATEGIES)}.")
        if any(asin not in self.asin_to_id for asin in history):
            raise ValueError("History contains an ASIN outside the available bundle catalog.")
        ids = [self.asin_to_id[asin] for asin in history]
        warm_ids = [item for item in ids if item < self.warm_items]
        branch_k = self.manifest["fusion"]["candidate_count"]
        with self.lock:
            content = []
            if strategy == "content" or strategy.endswith("_content"):
                content, _ = self.content.retrieve(
                    [{"history": ids, "seen_items": ids}],
                    top_k=branch_k,
                    max_history=self.manifest["max_history"],
                    batch_size=1,
                )
                content = content[0]
            ranked = []
            method = strategy.removesuffix("_content")
            if method in self.models and warm_ids:
                rows, _ = retrieve_rankings(
                    self.models[method],
                    [{"history": warm_ids, "seen_items": warm_ids}],
                    self.codebook,
                    max_history=self.manifest["max_history"],
                    top_k=branch_k,
                    beam_size=self.manifest["beam_size"],
                    batch_size=1,
                )
                ranked = rows[0]
            if strategy == "content":
                ranked = content
            elif strategy.endswith("_content"):
                ranked = reciprocal_rank_fusion(
                    ranked,
                    content,
                    top_k=branch_k,
                    content_weight=self.manifest["fusion"]["content_weight"],
                    rank_constant=self.manifest["fusion"]["rank_constant"],
                )
        return {
            "strategy": strategy,
            "serving_seed": self.manifest["serving_seed"],
            "warm_history_items": len(warm_ids),
            "available_history_items": len(ids),
            "recommendations": [
                {
                    "rank": rank,
                    "asin": self.catalog[item]["asin"],
                    "title": self.catalog[item]["title"],
                    "new_item": item >= self.warm_items,
                }
                for rank, item in enumerate(ranked[:top_k], 1)
            ],
        }


def make_server(engine: RecommendationEngine, host="127.0.0.1", port=8000):
    """Construct a local demo API. Caller owns serve_forever/shutdown lifecycle."""

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(15)

        def log_message(self, format, *args):
            # Request paths and payloads may contain user data; do not log them.
            pass

        def respond(self, status, payload):
            encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self):
            if self.path == "/health":
                self.respond(
                    200,
                    {
                        "status": "ok",
                        "strategies": list(STRATEGIES),
                        "warm_items": engine.warm_items,
                        "available_items": len(engine.catalog),
                        "diagnostic_only": engine.manifest["diagnostic_only"],
                    },
                )
            else:
                self.respond(404, {"error": "Unknown endpoint."})

        def do_POST(self):
            if self.path != "/recommend":
                self.respond(404, {"error": "Unknown endpoint."})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 65536:
                    self.respond(413, {"error": "JSON body must be 1–65536 bytes."})
                    return
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise ValueError("Incomplete request body.")
                body = json.loads(raw)
                if not isinstance(body, dict) or set(body) - {"history", "top_k", "strategy"}:
                    raise ValueError("Use a JSON object with history, top_k and strategy.")
                if "history" not in body:
                    raise ValueError("history is required.")
                self.respond(200, engine.recommend(**body))
            except (ValueError, TypeError, UnicodeError) as error:
                self.respond(400, {"error": str(error)})
            except Exception:
                self.respond(500, {"error": "Inference failed; inspect the local service."})

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server
