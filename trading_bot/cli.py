from __future__ import annotations

import argparse
import json
from pathlib import Path

from .backtest import run_research_pipeline, save_simulation_outputs, update_model_from_new_candles
from .config import FeeModel, ResearchConfig, StrategyParameters
from .data import load_candles


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Incremental crypto trading bot research runner.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Optimize, backtest, and save a trained bot state.")
    train_parser.add_argument(
        "--csv",
        type=Path,
        default=Path("bitcoin_price_history_five_minute_candlesticks.csv"),
        help="Path to the historical candle CSV.",
    )
    train_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory for metrics, trades, equity curves, and the saved model state.",
    )

    update_parser = subparsers.add_parser(
        "update-model",
        help="Incrementally update an existing saved state with newly arrived candle data.",
    )
    update_parser.add_argument("--state", type=Path, default=Path("artifacts/bot_state.joblib"))
    update_parser.add_argument("--csv", type=Path, required=True, help="CSV containing new candles to ingest.")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        candles = load_candles(args.csv)
        fee_model = FeeModel()
        research_config = ResearchConfig()
        strategy_parameters = StrategyParameters()
        research_result = run_research_pipeline(
            candles=candles,
            fee_model=fee_model,
            research_config=research_config,
            base_strategy_parameters=strategy_parameters,
        )
        best_validation = research_result["validation"]
        final_test = research_result["test"]
        output_paths = save_simulation_outputs(args.output_dir, final_test, research_config, fee_model)
        summary = {
            "validation_metrics": best_validation.metrics,
            "test_metrics": final_test.metrics,
            "best_strategy_parameters": final_test.strategy_parameters.to_dict(),
            "artifacts": {name: str(path) for name, path in output_paths.items()},
        }
        print(json.dumps(summary, indent=2))
        return

    if args.command == "update-model":
        update_summary = update_model_from_new_candles(args.state, args.csv)
        print(json.dumps(update_summary, indent=2))
        return

    raise ValueError(f"Unsupported command {args.command!r}")


if __name__ == "__main__":
    main()
