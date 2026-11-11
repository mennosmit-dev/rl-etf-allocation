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


def _fmt(rl: dict[str, float], ew: dict[str, float]) -> str:
    keys = ["cagr", "volatility", "sharpe", "max_drawdown"]
    rows = [f"  {'metric':<14}{'policy':>10}{'equal wt':>10}"]
    rows += [
        f"  {k:<14}{rl.get(k, float('nan')):>10.3f}{ew.get(k, float('nan')):>10.3f}" for k in keys
    ]
    return "\n".join(rows)


COMMANDS: dict[str, tuple[Callable[[Config, argparse.Namespace], None], str]] = {
    "data": (cmd_data, "build and validate the weekly return dataset"),
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
