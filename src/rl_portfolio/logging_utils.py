import logging
import os


def configure_logging(level: str | None = None) -> None:
    """Configure root logging once; level from argument or LOG_LEVEL env var."""
    level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for noisy in ("mlflow", "alembic", "urllib3", "yfinance"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
