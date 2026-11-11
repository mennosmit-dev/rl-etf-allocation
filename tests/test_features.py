import numpy as np
import pandas as pd
import pytest

from rl_portfolio.config import FeatureSpec
from rl_portfolio.features import available_features, build_feature_tensor, make_builders


@pytest.fixture
def returns() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.normal(0.001, 0.02, size=(80, 3)), columns=["A", "B", "C"])


def test_standard_features_registered():
    assert {"trailing_return", "rolling_volatility", "momentum"} <= set(available_features())


def test_unknown_feature_raises():
    with pytest.raises(KeyError):
        make_builders([FeatureSpec(name="does_not_exist", window=3)])


def test_tensor_shape_and_warmup(returns):
    specs = [FeatureSpec(name="trailing_return", window=4), FeatureSpec(name="momentum", window=10)]
    tensor = build_feature_tensor(returns, make_builders(specs))
    assert tensor.shape == (80, 2 * 3)
    assert np.isnan(tensor[:9]).any(axis=1).all()
    assert not np.isnan(tensor[9:]).any()


def test_trailing_return_compounds(returns):
    (builder,) = make_builders([FeatureSpec(name="trailing_return", window=3)])
    value = builder.compute(returns).iloc[10, 0]
    expected = np.prod(1 + returns.iloc[8:11, 0]) - 1
    assert value == pytest.approx(expected)


@pytest.mark.parametrize("name", ["trailing_return", "rolling_volatility", "momentum"])
def test_features_do_not_look_ahead(returns, name):
    """Changing a future return must not change any earlier feature value."""
    (builder,) = make_builders([FeatureSpec(name=name, window=5)])
    base = builder.compute(returns)
    shocked = returns.copy()
    shocked.iloc[50:] += 0.3
    after = builder.compute(shocked)
    pd.testing.assert_frame_equal(base.iloc[:50], after.iloc[:50])
