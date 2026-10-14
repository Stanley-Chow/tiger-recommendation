"""Exercise content controls, the full offline study, and a real local API."""

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from beautyrec.content_retrieval import ContentRetriever, reciprocal_rank_fusion
from beautyrec.m5_benchmark import run_m5
from beautyrec.m5_demo import run_demo
from beautyrec.serving import RecommendationEngine, make_server, verify_bundle


def test_content_retrieval_excludes_full_history_and_never_reads_target():
    vectors = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.float32)
    retriever = ContentRetriever(vectors)
    contexts = [{"history": [0], "seen_items": [0, 1], "target": 3}]
    first, _ = retriever.retrieve(contexts, top_k=4)
    contexts[0]["target"] = 2
    assert retriever.retrieve(contexts, top_k=4)[0] == first == [[2, 3]]
    with pytest.raises(ValueError):
        retriever.retrieve([{"history": [4]}])


def test_rank_fusion_is_deterministic_and_can_introduce_new_items():
    assert reciprocal_rank_fusion([0, 2], [1, 2]) == [2, 0, 1]
    assert reciprocal_rank_fusion([], [4, 3]) == [4, 3]
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([0, 0], [1])
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([0], [1], content_weight=float("nan"))


def test_offline_study_bundle_and_http_api(tmp_path):
    directory = tmp_path / "demo"
    root = Path(__file__).resolve().parents[1]
    result = run_demo(directory, root / "configs/m5.json")
    assert result["diagnostic_only"] and not result["m5_complete"]
    assert result["aggregate"]["mixed_new"]["generative"]["recall@200"]["mean"] == 0
    assert result["aggregate"]["mixed_new"]["content"]["recall@200"]["mean"] == 1
    config = json.loads((directory / "demo_config.json").read_text("utf-8"))
    # A completed run is checksum-verified and reused rather than retrained.
    reused = run_m5(
        directory / "processed", directory / "features", directory / "run", config, "cpu"
    )
    assert reused == result
    bundle = directory / "run" / "bundle"
    engine = RecommendationEngine(bundle)
    for strategy in ("generative", "sasrec", "content", "generative_content", "sasrec_content"):
        response = engine.recommend(["W0", "W1"], 5, strategy)
        assert response["recommendations"]
        assert not {"W0", "W1"} & {r["asin"] for r in response["recommendations"]}
    assert engine.recommend(["N0"], strategy="generative")["recommendations"] == []
    assert engine.recommend(["N0"], strategy="generative_content")["recommendations"]
    for payload in (["unknown"], [], ["W0", "W0"]):
        with pytest.raises(ValueError):
            engine.recommend(payload)
    server = make_server(engine, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(url + "/health", timeout=10) as response:
            assert json.load(response)["available_items"] == 10
        request = urllib.request.Request(
            url + "/recommend",
            method="POST",
            data=json.dumps({"history": ["W0"], "top_k": 3}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            assert len(json.load(response)["recommendations"]) == 3
        request = urllib.request.Request(url + "/recommend", method="POST", data=b'{"history":[]}')
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request, timeout=10)
        assert error.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    with (bundle / "catalog.json").open("a", encoding="utf-8") as file:
        file.write(" ")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_bundle(bundle)
