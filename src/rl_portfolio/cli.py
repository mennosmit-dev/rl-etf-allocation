"""Command line entry point: `uv run rl-etf <command> --config configs/example.yaml`."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable

from rl_portfolio.config import Config, load_config
from rl_portfolio.logging_utils import configure_logging

logger = logging.getLogger("rl_portfolio")


def cmd_data(config: Config, _args: argparse.Namespace) -> None:
    from rl_portfolio.data.dataset import build_dataset, save_dataset

    returns, meta = build_dataset(config)
    save_dataset(returns, meta, config.data.data_dir)


def cmd_train(config: Config, _args: argparse.Namespace) -> None:
    from rl_portfolio.training.pipeline import train_and_register

    record = train_and_register(config)
    logger.info(
        "Candidate %s\n%s",
        record.model_id,
        _fmt(record.validation_metrics, record.equal_weight_metrics),
    )


def cmd_retrain(config: Config, _args: argparse.Namespace) -> None:
    from rl_portfolio.training.pipeline import rolling_retrain

    result = rolling_retrain(config)
    logger.info(
        "Walk-forward out of sample over %d windows\n%s",
        len(result.windows),
        _fmt(result.rl_metrics, result.equal_weight_metrics),
    )
    logger.info("Candidate %s", result.candidate.model_id)


def cmd_promote(config: Config, args: argparse.Namespace) -> None:
    from rl_portfolio.registry import EvaluationGate, ModelRegistry

    registry = ModelRegistry(config.registry.path)
    model_id = args.model_id
    if model_id is None:
        candidates = [r for r in registry.list() if r.status == "candidate"]
        if not candidates:
            raise SystemExit("no candidate to promote")
        model_id = candidates[-1].model_id
    gate = EvaluationGate(min_sharpe_edge=config.registry.min_sharpe_edge)
    registry.promote(model_id, gate, reason=args.reason)


def cmd_rollback(config: Config, args: argparse.Namespace) -> None:
    from rl_portfolio.registry import ModelRegistry

    ModelRegistry(config.registry.path).rollback(reason=args.reason)


def cmd_serve(config: Config, args: argparse.Namespace) -> None:
    import uvicorn

    from rl_portfolio.serving.api import create_app

    uvicorn.run(create_app(config.registry.path), host=args.host, port=args.port)


def _fmt(rl: dict[str, float], ew: dict[str, float]) -> str:
    keys = ["cagr", "volatility", "sharpe", "max_drawdown"]
    rows = [f"  {'metric':<14}{'policy':>10}{'equal wt':>10}"]
    rows += [
        f"  {k:<14}{rl.get(k, float('nan')):>10.3f}{ew.get(k, float('nan')):>10.3f}" for k in keys
    ]
    return "\n".join(rows)


COMMANDS: dict[str, tuple[Callable[[Config, argparse.Namespace], None], str]] = {
    "data": (cmd_data, "build and validate the weekly return dataset"),
    "train": (cmd_train, "train on the latest window and register a candidate"),
    "retrain": (cmd_retrain, "walk-forward retrain and register the last window as candidate"),
    "promote": (cmd_promote, "promote a candidate through the evaluation gate"),
    "rollback": (cmd_rollback, "re-activate the previously active model"),
    "serve": (cmd_serve, "serve allocations over HTTP"),
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rl-etf")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, (_, help_text) in COMMANDS.items():
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--config", default="configs/example.yaml")
        p.add_argument("--log-level", default=None)
        if name == "train":
            p.add_argument(
                "--timesteps", type=int, default=None, help="override ppo.total_timesteps"
            )
        if name == "promote":
            p.add_argument("--model-id", default=None)
        if name in ("promote", "rollback"):
            p.add_argument("--reason", default="")
        if name in ("paper", "monitor"):
            p.add_argument("--weeks", type=int, default=52)
        if name == "serve":
            p.add_argument("--host", default="127.0.0.1")
            p.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    configure_logging(args.log_level)
    overrides = {}
    if getattr(args, "timesteps", None):
        overrides = {"ppo": {"total_timesteps": args.timesteps}}
    config = load_config(args.config, overrides)
    COMMANDS[args.command][0](config, args)


if __name__ == "__main__":
    main()
