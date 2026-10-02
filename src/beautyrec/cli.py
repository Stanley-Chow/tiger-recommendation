"""Explicit command-line stages: download, prepare, embed, tokenize, train, evaluate."""

import argparse
import json
import sys
from pathlib import Path

import torch

from beautyrec.data import create_demo_data, download_beauty, prepare_dataset
from beautyrec.decoding import SemanticTrie, constrained_beam_search
from beautyrec.embeddings import build_embeddings
from beautyrec.model import build_source_tokens
from beautyrec.semantic_ids import build_semantic_ids
from beautyrec.training import (
    TrainingConfig,
    check_catalog,
    check_dataset,
    evaluate_model,
    load_checkpoint,
    read_examples,
    resolve_device,
    sample_examples,
    train_model,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    download = commands.add_parser("download", help="Download original Amazon 2014 Beauty files")
    download.add_argument("--raw-dir", type=Path, default=Path("data/raw"))

    prepare = commands.add_parser("prepare", help="Build chronological examples and a warm catalog")
    prepare.add_argument("--reviews", type=Path, default=Path("data/raw/reviews_Beauty_5.json.gz"))
    prepare.add_argument("--metadata", type=Path, default=Path("data/raw/meta_Beauty.json.gz"))
    prepare.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    prepare.add_argument("--max-history", type=int, default=20)
    prepare.add_argument("--min-interactions", type=int, default=5)
    prepare.add_argument("--max-users", type=int)
    prepare.add_argument("--seed", type=int, default=42)

    embed = commands.add_parser("embed", help="Encode static product metadata")
    embed.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    embed.add_argument("--artifact-dir", type=Path, default=Path("artifacts/beauty"))
    embed.add_argument(
        "--backend", choices=["sentence-transformers", "tfidf"], default="sentence-transformers"
    )
    embed.add_argument("--model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    embed.add_argument("--batch-size", type=int, default=64)
    embed.add_argument("--device", default="cpu")
    embed.add_argument("--seed", type=int, default=42)
    embed.add_argument(
        "--max-text-length",
        type=int,
        help="Optional token limit below the encoder's native limit, for faster CPU encoding",
    )

    tokenize = commands.add_parser("tokenize", help="Fit hierarchical k-means semantic IDs")
    tokenize.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    tokenize.add_argument("--artifact-dir", type=Path, default=Path("artifacts/beauty"))
    tokenize.add_argument("--branching-factor", type=int, default=16)
    tokenize.add_argument("--depth", type=int, default=3)
    tokenize.add_argument("--seed", type=int, default=42)

    train = commands.add_parser("train", help="Train and save the best validation checkpoint")
    train.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    train.add_argument("--artifact-dir", type=Path, default=Path("artifacts/beauty"))
    train.add_argument("--run-dir", type=Path, default=Path("results/beauty"))
    train.add_argument("--config", type=Path, default=Path("configs/beauty.json"))
    train.add_argument("--device", help="Override the config device (auto, cpu, cuda)")

    evaluate = commands.add_parser("evaluate", help="Compare retrieval with popularity")
    evaluate.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    evaluate.add_argument("--artifact-dir", type=Path, default=Path("artifacts/beauty"))
    evaluate.add_argument("--checkpoint", type=Path, default=Path("results/beauty/best.pt"))
    evaluate.add_argument("--output-dir", type=Path, default=Path("results/beauty/evaluation"))
    evaluate.add_argument("--split", choices=["validation", "test"], default="test")
    evaluate.add_argument("--top-k", type=int, default=10)
    evaluate.add_argument("--beam-size", type=int, default=20)
    evaluate.add_argument("--max-examples", type=int)
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument("--seed", type=int, default=42)
    evaluate.add_argument(
        "--cutoffs",
        nargs="+",
        type=int,
        help="Evaluate these cutoffs from one candidate list, including MAP/MRR",
    )
    evaluate.add_argument("--batch-size", type=int, default=8)

    benchmark = commands.add_parser("benchmark", help="Run M1/M2 baselines and SAM comparisons")
    benchmark.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    benchmark.add_argument("--artifact-dir", type=Path, default=Path("artifacts/beauty"))
    benchmark.add_argument("--run-dir", type=Path, default=Path("results/m2"))
    benchmark.add_argument("--config", type=Path, default=Path("configs/m2.json"))
    benchmark.add_argument("--device", help="Override the experiment device (auto, cpu, cuda)")

    recommend = commands.add_parser("recommend", help="Generate products from an ASIN history")
    recommend.add_argument("--history", nargs="+", required=True, help="Oldest to newest ASINs")
    recommend.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    recommend.add_argument("--artifact-dir", type=Path, default=Path("artifacts/beauty"))
    recommend.add_argument("--checkpoint", type=Path, default=Path("results/beauty/best.pt"))
    recommend.add_argument("--top-k", type=int, default=10)
    recommend.add_argument("--beam-size", type=int, default=20)
    recommend.add_argument("--device", default="auto")

    demo = commands.add_parser("demo", help="Run a small synthetic, offline pipeline check")
    demo.add_argument("--work-dir", type=Path, default=Path("results/demo"))
    demo.add_argument("--epochs", type=int, default=3)
    return parser


def run_evaluation(args: argparse.Namespace) -> dict:
    import numpy as np

    device = resolve_device(args.device)
    torch.set_num_threads(4)
    cutoffs = getattr(args, "cutoffs", None)
    if cutoffs:
        from beautyrec.experiment_training import load_experiment_checkpoint

        model, codebook, checkpoint = load_experiment_checkpoint(
            args.checkpoint, args.artifact_dir, device
        )
    else:
        model, codebook, checkpoint = load_checkpoint(args.checkpoint, args.artifact_dir, device)
    check_catalog(args.processed_dir, codebook)
    check_dataset(args.processed_dir, checkpoint)
    if args.max_examples is not None and args.max_examples < 1:
        raise ValueError("--max-examples must be positive.")
    examples = sample_examples(
        read_examples(args.processed_dir / f"{args.split}.jsonl"),
        args.max_examples,
        args.seed,
    )
    popularity = np.load(args.processed_dir / "item_popularity.npy", allow_pickle=False)
    if cutoffs:
        from beautyrec.evaluation import evaluate_rankings
        from beautyrec.metrics import validate_cutoffs
        from beautyrec.retrieval import popularity_rankings, retrieve_rankings

        cutoffs = validate_cutoffs(cutoffs)
        maximum = max(cutoffs)
        rankings, timing = retrieve_rankings(
            model,
            examples,
            codebook,
            max_history=checkpoint["training_config"]["max_history"],
            top_k=maximum,
            beam_size=args.beam_size,
            batch_size=args.batch_size,
            progress=True,
        )
        method = checkpoint.get("method", "generative")
        summary = evaluate_rankings(
            examples,
            {method: rankings, "popularity": popularity_rankings(examples, popularity, maximum)},
            popularity,
            cutoffs=cutoffs,
            output_dir=args.output_dir,
        )
        summary["retrieval_timing"] = timing
        summary["beam_size"] = args.beam_size
        summary["embedding_backend"] = codebook.diagnostics.get("embedding_backend")
        summary["diagnostic_only"] = codebook.diagnostics.get("diagnostic_only", False)
    else:
        summary = evaluate_model(
            model,
            examples,
            codebook,
            popularity,
            checkpoint["training_config"]["max_history"],
            args.top_k,
            args.beam_size,
            args.output_dir / "per_user_metrics.csv",
        )
    summary.update(
        {
            "split": args.split,
            "sample_limit": args.max_examples,
            "sampling_seed": args.seed,
            "checkpoint_epoch": checkpoint["epoch"],
        }
    )
    write_json(args.output_dir / "summary.json", summary)
    return summary


def run_recommendation(args: argparse.Namespace) -> list[dict]:
    torch.set_num_threads(4)
    model, codebook, checkpoint = load_checkpoint(
        args.checkpoint,
        args.artifact_dir,
        resolve_device(args.device),
    )
    check_catalog(args.processed_dir, codebook)
    check_dataset(args.processed_dir, checkpoint)
    catalog = json.loads((args.processed_dir / "catalog.json").read_text(encoding="utf-8"))
    mapping = {item["asin"]: item["item_id"] for item in catalog}
    unknown = [asin for asin in args.history if asin not in mapping]
    if unknown:
        raise ValueError(f"Unknown ASINs: {unknown}. Use products from this training catalog.")
    history = [mapping[asin] for asin in args.history]
    source = torch.tensor(
        build_source_tokens(
            history, codebook.item_codes, checkpoint["training_config"]["max_history"]
        ),
        dtype=torch.long,
        device=next(model.parameters()).device,
    )
    results = constrained_beam_search(
        model,
        source,
        SemanticTrie(codebook.item_codes),
        args.top_k,
        args.beam_size,
        set(history),
    )
    return [
        {"asin": catalog[item]["asin"], "title": catalog[item]["title"], "log_probability": score}
        for item, score in results
    ]


def run_demo(work_dir: Path, epochs: int) -> dict:
    """A functional check using invented data; its metrics are not Beauty results."""
    raw, processed, artifacts = work_dir / "raw", work_dir / "processed", work_dir / "artifacts"
    reviews, metadata = create_demo_data(raw)
    statistics = prepare_dataset(reviews, metadata, processed, max_history=5)
    build_embeddings(processed, artifacts, backend="tfidf")
    build_semantic_ids(processed, artifacts, branching_factor=2, depth=2)
    train_model(
        processed,
        artifacts,
        work_dir / "training",
        TrainingConfig(
            epochs=epochs,
            batch_size=16,
            max_history=5,
            patience=epochs,
            eval_max_examples=None,
            device="cpu",
            top_k=5,
            beam_size=10,
        ),
        {
            "d_model": 32,
            "nhead": 4,
            "num_encoder_layers": 1,
            "num_decoder_layers": 1,
            "dim_feedforward": 64,
            "dropout": 0.0,
        },
    )
    summary = run_evaluation(
        argparse.Namespace(
            processed_dir=processed,
            artifact_dir=artifacts,
            checkpoint=work_dir / "training/best.pt",
            output_dir=work_dir / "evaluation",
            split="test",
            top_k=5,
            beam_size=10,
            max_examples=None,
            device="cpu",
            seed=42,
        )
    )
    summary["diagnostic_only"] = True
    summary["dataset"] = "synthetic demo, not Amazon Beauty"
    write_json(work_dir / "evaluation/summary.json", summary)
    return {"data": statistics, "evaluation": summary}


def main() -> None:
    # Windows' legacy console encoding cannot represent every Amazon product title.
    # UTF-8 also gives redirected JSON a consistent encoding on all platforms.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.command == "download":
            result = {name: str(path) for name, path in download_beauty(args.raw_dir).items()}
        elif args.command == "prepare":
            result = prepare_dataset(
                args.reviews,
                args.metadata,
                args.processed_dir,
                max_history=args.max_history,
                min_interactions=args.min_interactions,
                max_users=args.max_users,
                seed=args.seed,
            )
            # Full selected user IDs remain on disk for reproducibility, not in console output.
            result = {key: value for key, value in result.items() if key != "selected_user_ids"}
        elif args.command == "embed":
            torch.set_num_threads(4)
            result = build_embeddings(
                args.processed_dir,
                args.artifact_dir,
                backend=args.backend,
                model_name=args.model_name,
                batch_size=args.batch_size,
                device=args.device,
                seed=args.seed,
                max_text_length=args.max_text_length,
            )
        elif args.command == "tokenize":
            result = build_semantic_ids(
                args.processed_dir,
                args.artifact_dir,
                branching_factor=args.branching_factor,
                depth=args.depth,
                seed=args.seed,
            )
            result = {key: value for key, value in result.items() if key != "item_codes"}
        elif args.command == "train":
            config = json.loads(args.config.read_text(encoding="utf-8"))
            options = dict(config.get("training", {}))
            if args.device:
                options["device"] = args.device
            result = train_model(
                args.processed_dir,
                args.artifact_dir,
                args.run_dir,
                TrainingConfig(**options),
                config.get("model", {}),
            )
        elif args.command == "evaluate":
            result = run_evaluation(args)
        elif args.command == "recommend":
            result = run_recommendation(args)
        elif args.command == "benchmark":
            from beautyrec.benchmark import run_benchmark

            config = json.loads(args.config.read_text(encoding="utf-8"))
            result = run_benchmark(
                args.processed_dir,
                args.artifact_dir,
                args.run_dir,
                config,
                device_override=args.device,
            )
        else:
            result = run_demo(args.work_dir, args.epochs)
    except (ValueError, FileNotFoundError, RuntimeError, ImportError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
