# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for the golden-output test helper."""

import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.base import BaseForecaster
from sktime.tests.test_switch import run_test_module_changed
from sktime.utils._testing.golden import GoldenCase, assert_golden_forecast

pytestmark = pytest.mark.skipif(
    not run_test_module_changed("sktime.utils._testing.golden"),
    reason="run test only if the golden test helper has changed",
)


class _StepForecaster(BaseForecaster):
    """Forecasts ``last value + h`` at step ``h``, quantiles shifted by alpha.

    The quantile forecast at level ``alpha`` is the point forecast plus
    ``10 * (alpha - 0.5) + offset``.
    """

    _tags = {
        "y_inner_mtype": "pd.DataFrame",
        "capability:multivariate": True,
        "capability:pred_int": True,
        "requires-fh-in-fit": False,
    }

    def __init__(self, offset=0.0):
        self.offset = offset
        super().__init__()

    def _fit(self, y, X, fh):
        self.last_ = y.iloc[-1]
        return self

    def _predict(self, fh, X):
        steps = np.asarray(fh.to_relative(self.cutoff), dtype=float)
        values = self.last_.to_numpy()[None, :] + steps[:, None]
        index = fh.to_absolute_index(self.cutoff)
        return pd.DataFrame(values, index=index, columns=self._y.columns)

    def _predict_quantiles(self, fh, X, alpha):
        y_pred = self._predict(fh, X)
        columns = pd.MultiIndex.from_product([y_pred.columns, alpha])
        values = np.concatenate(
            [
                y_pred.to_numpy()[:, [i]] + 10 * (a - 0.5) + self.offset
                for i in range(y_pred.shape[1])
                for a in alpha
            ],
            axis=1,
        )
        return pd.DataFrame(values, index=y_pred.index, columns=columns)


class _HorizonDependentForecaster(_StepForecaster):
    """Like ``_StepForecaster``, with quantiles shifted by the largest step in fh.

    Mimics autoregressive models, whose output at a step can depend on how far
    ahead the forecast is generated.
    """

    def _predict_quantiles(self, fh, X, alpha):
        pred_quantiles = super()._predict_quantiles(fh, X, alpha)
        return pred_quantiles + max(fh.to_relative(self.cutoff))


def _univariate():
    return pd.Series(
        [1.0, 2.0, 3.0, 4.0],
        index=pd.period_range("2000-01", periods=4, freq="M"),
        name="y",
    )


def _multivariate():
    return pd.DataFrame(
        {"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]},
        index=pd.period_range("2000-01", periods=3, freq="M"),
    )


def test_golden_case_passes():
    """A case matching the forecasts passes, also with a non-contiguous fh."""
    case = GoldenCase(
        name="univariate",
        estimator_params={},
        fixture=_univariate,
        fh=[1, 2, 10],
        expected=np.array([5.0, 6.0, 14.0]),
    )
    assert_golden_forecast(_StepForecaster, case)


def test_golden_case_failure_message():
    """A mismatch fails with the case name and the max abs and rel diffs."""
    case = GoldenCase(
        name="off-by-some",
        estimator_params={},
        fixture=_univariate,
        fh=[1, 2, 3],
        expected=np.array([5.0, 6.0, 8.0]),
        provenance="made up for this test",
    )
    with pytest.raises(AssertionError) as excinfo:
        assert_golden_forecast(_StepForecaster, case)

    msg = str(excinfo.value)
    assert "golden case 'off-by-some': point forecast" in msg
    assert "max abs diff 1," in msg
    assert "max rel diff 0.125" in msg
    assert "reference provenance: made up for this test" in msg


def test_golden_case_shape_mismatch_message():
    """A reference of the wrong shape fails with both shapes in the message."""
    case = GoldenCase(
        name="short",
        estimator_params={},
        fixture=_univariate,
        fh=[1, 2, 3],
        expected=np.array([5.0, 6.0]),
    )
    with pytest.raises(AssertionError, match=r"has shape \(3,\), expected \(2,\)"):
        assert_golden_forecast(_StepForecaster, case)


def test_golden_case_tolerances():
    """Differences within atol pass, differences beyond it fail."""
    kwargs = {
        "estimator_params": {},
        "fixture": _univariate,
        "fh": [1],
        "rtol": 0.0,
        "atol": 1e-2,
    }
    ok = GoldenCase(name="within", expected=np.array([5.005]), **kwargs)
    assert_golden_forecast(_StepForecaster, ok)

    bad = GoldenCase(name="beyond", expected=np.array([5.02]), **kwargs)
    with pytest.raises(AssertionError, match="golden case 'beyond'"):
        assert_golden_forecast(_StepForecaster, bad)


def test_golden_case_quantiles():
    """Quantile references are checked, and a mismatch names the quantile."""
    expected_quantiles = {
        0.1: np.array([1.0, 2.0]),
        0.9: np.array([9.0, 10.0]),
    }
    case = GoldenCase(
        name="quantiles",
        estimator_params={},
        fixture=_univariate,
        fh=[1, 2],
        expected=np.array([5.0, 6.0]),
        expected_quantiles=expected_quantiles,
    )
    assert_golden_forecast(_StepForecaster, case)

    shifted = GoldenCase(
        name="quantiles-shifted",
        estimator_params={"offset": 1.0},
        fixture=_univariate,
        fh=[1, 2],
        expected=np.array([5.0, 6.0]),
        expected_quantiles=expected_quantiles,
    )
    msg = r"golden case 'quantiles-shifted': quantile 0.1 forecast"
    with pytest.raises(AssertionError, match=msg):
        assert_golden_forecast(_StepForecaster, shifted)


def test_golden_case_multivariate():
    """Multivariate references are 2D, one column per variable."""
    case = GoldenCase(
        name="multivariate",
        estimator_params={},
        fixture=_multivariate,
        fh=[1, 2],
        expected=np.array([[4.0, 31.0], [5.0, 32.0]]),
        expected_quantiles={0.5: np.array([[4.0, 31.0], [5.0, 32.0]])},
    )
    assert_golden_forecast(_StepForecaster, case)

    transposed = GoldenCase(
        name="multivariate-transposed",
        estimator_params={},
        fixture=_multivariate,
        fh=[1, 2],
        expected=np.array([[4.0, 5.0], [31.0, 32.0]]),
    )
    with pytest.raises(AssertionError, match="golden case 'multivariate-transposed'"):
        assert_golden_forecast(_StepForecaster, transposed)


def test_golden_case_quantile_fh():
    """Quantiles are generated at the full fh and compared at quantile_fh only."""
    case = GoldenCase(
        name="quantile-sub-horizon",
        estimator_params={},
        fixture=_univariate,
        fh=[1, 2, 10],
        expected=np.array([5.0, 6.0, 14.0]),
        # point forecast at steps 1 and 2, shifted by the largest step 10
        expected_quantiles={0.5: np.array([15.0, 16.0])},
        quantile_fh=[1, 2],
    )
    assert_golden_forecast(_HorizonDependentForecaster, case)

    outside = GoldenCase(
        name="quantile-outside-fh",
        estimator_params={},
        fixture=_univariate,
        fh=[1, 2],
        expected=np.array([5.0, 6.0]),
        expected_quantiles={0.5: np.array([7.0])},
        quantile_fh=[3],
    )
    with pytest.raises(AssertionError, match="quantile_fh steps .* are not in fh"):
        assert_golden_forecast(_HorizonDependentForecaster, outside)
