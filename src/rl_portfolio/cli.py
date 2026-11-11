"""Command line entry point: `uv run rl-etf <command> --config configs/example.yaml`."""

from __future__ import annotations

import argparse
import json
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


def cmd_paper(config: Config, args: argparse.Namespace) -> None:
    from rl_portfolio.data.dataset import get_or_build_dataset
    from rl_portfolio.envs.sizing import make_sizer
    from rl_portfolio.paper.simulate import equal_weight_fn, run_paper_trading
    from rl_portfolio.registry import ModelRegistry
    from rl_portfolio.serving.inference import AllocationPolicy

    record = ModelRegistry(config.registry.path).active()
    if record is None:
        raise SystemExit("no active model; run `rl-etf promote` first")
    policy = AllocationPolicy(record)
    returns, _ = get_or_build_dataset(config)
    start = len(returns) - args.weeks
    common = dict(
        returns=returns,
        start=start,
        sizer=make_sizer(config.sizing),
        initial_cash=config.sizing.capital,
        cost_rate=config.reward.transaction_cost,
        max_turnover=config.monitoring.max_turnover,
    )
    rl = run_paper_trading(policy.target_weights, **common)
    ew = run_paper_trading(equal_weight_fn(record.tickers), **common)
    logger.info(
        "Paper trading over the last %d weeks (simulated broker)\n%s",
        args.weeks,
        _fmt(rl.metrics, ew.metrics),
    )


def cmd_monitor(config: Config, args: argparse.Namespace) -> None:
    from rl_portfolio.data.dataset import get_or_build_dataset
    from rl_portfolio.monitoring.drift import check_drift
    from rl_portfolio.registry import ModelRegistry

    record = ModelRegistry(config.registry.path).active()
    if record is None:
        raise SystemExit("no active model")
    returns, _ = get_or_build_dataset(config)
    returns = returns[record.tickers]
    dates = list(map(str, returns.index))
    lo, hi = dates.index(record.train_window[0]), dates.index(record.train_window[1])
    # Weekly returns, not the rolling features: overlapping windows make a few
    # months of a 26-week feature nearly one value, which inflates PSI.
    values = returns.to_numpy()
    report = check_drift(
        values[lo : hi + 1],
        values[-args.weeks :],
        [f"return:{t}" for t in record.tickers],
        psi_threshold=config.monitoring.psi_threshold,
    )
    worst = sorted(report.feature_psi.items(), key=lambda kv: -kv[1])[:5]
    logger.info(
        "Drift check over the last %d weeks: %s; highest PSI %s",
        args.weeks,
        "ok" if report.ok else f"drift in {report.drifted_features}",
        json.dumps({k: round(v, 3) for k, v in worst}),
    )


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
    "paper": (cmd_paper, "paper-trade the active model against equal weight"),
    "monitor": (cmd_monitor, "return drift of recent weeks against the training window"),
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
