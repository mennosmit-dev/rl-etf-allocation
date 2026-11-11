import numpy as np
import pytest

from rl_portfolio.data.dataset import (
    build_dataset,
    content_hash,
    get_or_build_dataset,
    load_dataset,
    save_dataset,
)
from rl_portfolio.data.sources import synthetic_prices
from rl_portfolio.data.validation import DataValidationError, validate_prices

TICKERS = ["AAA", "BBB", "CCC"]


def test_synthetic_prices_are_reproducible():
    a = synthetic_prices(TICKERS, weeks=100, seed=1)
    b = synthetic_prices(TICKERS, weeks=100, seed=1)
    assert a.equals(b)
    assert list(a.columns) == ["Date", *TICKERS]
    assert (a[TICKERS] > 0).all().all()


def test_validation_accepts_clean_prices():
    prices = synthetic_prices(TICKERS, weeks=100, seed=1)
    assert len(validate_prices(prices, TICKERS)) == 100


@pytest.mark.parametrize(
    "corrupt, message",
    [
        (lambda df: df.drop(columns="BBB"), "missing columns"),
        (lambda df: df.assign(AAA=df["AAA"].where(df.index != 10, -1.0)), "schema"),
        (lambda df: df.assign(AAA=df["AAA"].where(df.index != 10, np.nan)), "schema"),
        (lambda df: df.iloc[::-1], "sorted"),
        (lambda df: df.assign(CCC=df["CCC"].where(df.index != 50, df["CCC"] * 3)), "weekly moves"),
        (lambda df: df.head(30), "rows"),
    ],
)
def test_validation_rejects_bad_prices(corrupt, message):
    prices = corrupt(synthetic_prices(TICKERS, weeks=100, seed=1))
    with pytest.raises(DataValidationError, match=message):
        validate_prices(prices, TICKERS)


def test_dataset_roundtrip_and_hash(config):
    returns, meta = build_dataset(config)
    assert meta.row_count == len(returns) == config.data.synthetic_weeks - 1
    save_dataset(returns, meta, config.data.data_dir)
    loaded, loaded_meta = load_dataset(config.data.data_dir)
    assert loaded_meta == meta
    assert content_hash(loaded) == meta.content_hash


def test_tampered_dataset_is_detected(config):
    returns, meta = build_dataset(config)
    save_dataset(returns, meta, config.data.data_dir)
    returns.iloc[5, 0] += 0.01
    returns.to_parquet(config.data.data_dir / "weekly_returns.parquet")
    with pytest.raises(ValueError, match="hash"):
        load_dataset(config.data.data_dir)


def test_get_or_build_uses_cache(config):
    first, meta1 = get_or_build_dataset(config)
    second, meta2 = get_or_build_dataset(config)
    assert meta1.content_hash == meta2.content_hash
    assert first.equals(second)
