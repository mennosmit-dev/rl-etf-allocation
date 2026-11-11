FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.8.0 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --locked --no-dev --no-install-project

COPY src ./src
COPY configs ./configs
RUN uv sync --locked --no-dev

# One image, several scheduled jobs: the scheduler passes the subcommand.
ENTRYPOINT ["uv", "run", "--no-sync", "rl-etf"]
CMD ["--help"]
