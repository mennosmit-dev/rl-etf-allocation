.PHONY: install lint format test data train retrain promote paper monitor serve

CONFIG ?= configs/example.yaml

install:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .

test:
	uv run pytest -q

data:
	uv run rl-etf data --config $(CONFIG)

train:
	uv run rl-etf train --config $(CONFIG)

retrain:
	uv run rl-etf retrain --config $(CONFIG)

promote:
	uv run rl-etf promote --config $(CONFIG)

paper:
	uv run rl-etf paper --config $(CONFIG)

monitor:
	uv run rl-etf monitor --config $(CONFIG)

serve:
	uv run rl-etf serve --config $(CONFIG)
