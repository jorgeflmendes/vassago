"""Command-line entry points share the canonical Python experiment APIs."""

import argparse
import json
from pathlib import Path

from vassago.config import ExperimentConfig
from vassago.data import build_movielens
from vassago.experiment import run
from vassago.serving import RecommendationContext, UserPreferenceProfile, VassagoRanker


def main() -> None:
    parser = argparse.ArgumentParser(prog="vassago")
    commands = parser.add_subparsers(dest="command", required=True)
    plot = commands.add_parser("plot")
    plot.add_argument("--run", type=Path, required=True)
    plot.add_argument("--output", type=Path, required=True)
    tune = commands.add_parser("tune")
    tune.add_argument("--config", type=Path, required=True)
    tune.add_argument("--space", type=Path, required=True)
    tune.add_argument("--output", type=Path, required=True)
    tune.add_argument("--data", type=Path)
    experiment = commands.add_parser("experiment")
    experiment.add_argument("action", choices=["run"])
    experiment.add_argument("--config", type=Path, default=Path("configs/experiment/smoke.yaml"))
    experiment.add_argument("--output", type=Path, default=Path("reports/runs"))
    experiment.add_argument("--data", type=Path)
    train = commands.add_parser("train")
    train.add_argument("--config", type=Path, default=Path("configs/experiment/smoke.yaml"))
    train.add_argument("--output", type=Path, default=Path("reports/runs"))
    train.add_argument("--data", type=Path)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--run", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    embeddings = commands.add_parser("embeddings")
    embeddings.add_argument("action", choices=["build"])
    embeddings.add_argument("--data", type=Path, required=True)
    embeddings.add_argument("--config", type=Path, required=True)
    embeddings.add_argument("--cutoff", type=int, required=True)
    embeddings.add_argument("--output", type=Path, required=True)
    data = commands.add_parser("data")
    data.add_argument("action", choices=["build", "download", "enrich"])
    data.add_argument("--source", type=Path)
    data.add_argument("--output", type=Path, default=Path("data/processed/movielens"))
    data.add_argument("--release", choices=["ml-32m", "ml-1m", "ml-100k"], default="ml-32m")
    data.add_argument("--metadata-available-at", type=int)
    recommend = commands.add_parser("recommend")
    recommend.add_argument("--model", type=Path, required=True)
    recommend.add_argument("--timestamp", type=int, required=True)
    recommend.add_argument("--genre", action="append", default=[])
    recommend.add_argument("--k", type=int, default=10)
    benchmark = commands.add_parser("benchmark")
    benchmark.add_argument(
        "action",
        choices=[
            "prepare",
            "contextual",
            "popularity",
            "evaluate",
            "freeze-selection",
        ],
    )
    benchmark.add_argument("--config", type=Path)
    benchmark.add_argument("--data", type=Path)
    benchmark.add_argument("--protocol", type=Path)
    benchmark.add_argument("--predictions", type=Path, nargs="+")
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--history-length", type=int, default=200)
    benchmark.add_argument("--global-test-start", type=int)
    benchmark.add_argument("--query-sample-limit", type=int)
    benchmark.add_argument("--query-sample-seed", type=int, default=42)
    benchmark.add_argument("--seed", type=int)
    benchmark.add_argument("--selection-recipe", type=Path)
    benchmark.add_argument("--selection-manifest", type=Path)
    benchmark.add_argument("--validation-only", action="store_true")
    args = parser.parse_args()
    if args.command == "benchmark" and args.validation_only and args.action != "contextual":
        parser.error("--validation-only requires benchmark contextual")
    if args.command == "plot":
        from vassago.reports import complementarity_plot

        complementarity_plot(args.run, args.output)
    elif args.command == "tune":
        from vassago.search import search

        print(
            search(
                ExperimentConfig.read(args.config),
                json.loads(args.space.read_text()),
                args.output,
                args.data,
            )
        )
    elif args.command in {"experiment", "train"}:
        print(run(ExperimentConfig.read(args.config), args.output, args.data))
    elif args.command == "evaluate":
        from vassago.pipeline import evaluate_checkpoint

        print(json.dumps(evaluate_checkpoint(args.run, args.output), indent=2))
    elif args.command == "embeddings":
        from vassago.pipeline import build_embeddings

        print(
            build_embeddings(
                args.data, ExperimentConfig.read(args.config), args.cutoff, args.output
            )
        )
    elif args.command == "data":
        if args.action == "download":
            from vassago.sources import download_movielens

            print(download_movielens(args.release, args.output))
        elif args.action == "enrich":
            from vassago.pipeline import enrich_catalog

            if args.source is None or args.metadata_available_at is None:
                parser.error("data enrich requires --source and --metadata-available-at")
            enrich_catalog(args.source, args.output, args.metadata_available_at)
        else:
            if args.source is None:
                parser.error("data build requires --source")
            build_movielens(args.source, args.output)
    elif args.command == "recommend":
        model = VassagoRanker.load(args.model)
        result = model.recommend(
            [],
            args.k,
            UserPreferenceProfile(genres=args.genre),
            RecommendationContext(timestamp=args.timestamp),
        )
        print(json.dumps([r.model_dump() for r in result], indent=2))
    elif args.command == "benchmark":
        from vassago.fair_benchmark import (
            evaluate_predictions,
            popularity_predictions,
            prepare_protocol,
        )

        if args.action == "prepare":
            if args.data is None:
                parser.error("benchmark prepare requires --data")
            print(
                prepare_protocol(
                    args.data,
                    args.output,
                    args.history_length,
                    args.global_test_start,
                    args.query_sample_limit,
                    args.query_sample_seed,
                )
            )
        elif args.action == "freeze-selection":
            if args.config is None or args.selection_manifest is None:
                parser.error(
                    "benchmark freeze-selection requires --config and --selection-manifest"
                )
            from vassago.contextual_ranker import freeze_selection_recipe

            print(
                freeze_selection_recipe(
                    args.selection_manifest, ExperimentConfig.read(args.config), args.output
                )
            )
        elif args.action == "contextual":
            if args.data is None or args.protocol is None or args.config is None:
                parser.error("benchmark contextual requires --data, --protocol and --config")
            from vassago.contextual_ranker import run_fair_contextual

            config = ExperimentConfig.read(args.config)
            if args.seed is not None:
                config = config.model_copy(update={"seed": args.seed})
            print(
                run_fair_contextual(
                    config,
                    args.data,
                    args.protocol,
                    args.output,
                    validation_only=args.validation_only,
                    selection_recipe=args.selection_recipe,
                )
            )
        elif args.action == "popularity":
            if args.data is None or args.protocol is None:
                parser.error("benchmark popularity requires --data and --protocol")
            seed = 42 if args.seed is None else args.seed
            print(popularity_predictions(args.data, args.protocol, seed, args.output))
        else:
            if args.protocol is None or args.predictions is None:
                parser.error("benchmark evaluate requires --protocol and --predictions")
            print(evaluate_predictions(args.protocol, args.predictions, args.output))


if __name__ == "__main__":
    main()
