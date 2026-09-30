# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Shared helper for golden-output regression tests of forecasters.

A golden-output test fits a forecaster on a fixed series and compares its
forecasts with reference values produced outside sktime, typically by the
upstream implementation that the forecaster wraps or vendors.

Each test module declares its reference values as a list of ``GoldenCase``
objects and checks them with ``assert_golden_forecast``, for example::

    CASES = [
        GoldenCase(
            name="default",
            estimator_params={"model_name": "some/checkpoint"},
            fixture=load_train_series,
            fh=[1, 2, 3],
            expected=np.array([1.0, 2.0, 3.0]),
            provenance="upstream some/checkpoint at commit abc123",
        ),
    ]

    @pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
    def test_matches_upstream_reference(case):
        assert_golden_forecast(MyForecaster, case)
"""

__all__ = ["GoldenCase", "assert_golden_forecast"]

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GoldenCase:
    """Reference forecasts of one estimator configuration on one series.

    Parameters
    ----------
    name : str
        Case name, used as pytest id and in failure messages.
    estimator_params : dict
        Keyword arguments passed to the estimator constructor.
    fixture : callable with no arguments, returning pd.Series or pd.DataFrame
        Returns the training series: a ``pd.Series`` for univariate cases,
        a ``pd.DataFrame`` for multivariate cases.
        It is called after the estimator is constructed and immediately before
        ``fit``, so it may also set random seeds that ``fit`` depends on.
    fh : int, list, np.ndarray or ForecastingHorizon
        Forecasting horizon passed to ``fit`` and ``predict``.
        Need not be contiguous, e.g., only the first and last few steps of a
        long horizon can be checked.
    expected : np.ndarray
        Expected point forecasts, of shape ``(len(fh),)`` for a univariate
        fixture, or ``(len(fh), n_variables)`` for a multivariate fixture.
    expected_quantiles : dict of float to np.ndarray, or None, default=None
        Expected quantile forecasts, keyed by quantile level ``alpha``,
        each of shape ``(len(quantile_fh),)`` for a univariate fixture, or
        ``(len(quantile_fh), n_variables)`` for a multivariate fixture.
        If None, ``predict_quantiles`` is not called.
    quantile_fh : int, list, np.ndarray, ForecastingHorizon or None, default=None
        Steps at which quantile forecasts are compared, a subset of ``fh``.
        ``predict_quantiles`` is always called with the full ``fh``, so
        quantiles are generated exactly as for the point forecast, and only
        the rows at ``quantile_fh`` are compared. If None, ``fh`` is used.
    rtol : float, default=1e-5
        Relative tolerance, passed to ``np.testing.assert_allclose``.
    atol : float, default=1e-4
        Absolute tolerance, passed to ``np.testing.assert_allclose``.
    provenance : str, default=""
        How the reference values were produced, e.g., upstream repository,
        commit and environment.
    """

    name: str
    estimator_params: dict
    fixture: Callable[[], pd.Series | pd.DataFrame]
    fh: object
    expected: np.ndarray
    expected_quantiles: dict[float, np.ndarray] | None = None
    quantile_fh: object = None
    rtol: float = 1e-5
    atol: float = 1e-4
    provenance: str = ""


def assert_golden_forecast(estimator_cls, case):
    """Assert that an estimator reproduces the reference forecasts of a case.

    Constructs ``estimator_cls(**case.estimator_params)``, fits it on
    ``case.fixture()`` with ``fh=case.fh``, and compares ``predict`` with
    ``case.expected``. If ``case.expected_quantiles`` is set, also compares
    ``predict_quantiles`` at those quantile levels, over the full ``fh``,
    at the steps in ``case.quantile_fh``.

    Parameters
    ----------
    estimator_cls : sktime forecaster class
        Forecaster class to test.
    case : GoldenCase
        Reference case to check.

    Raises
    ------
    AssertionError
        If any forecast differs in shape from, or is not close to, its
        reference. The message names the case and reports the maximum
        absolute and relative differences.
    """
    estimator = estimator_cls(**case.estimator_params)
    y_train = case.fixture()
    estimator.fit(y_train, fh=case.fh)

    y_pred = estimator.predict(fh=case.fh)
    _assert_close(case, "point forecast", y_pred, case.expected)

    if case.expected_quantiles is None:
        return

    alpha = sorted(case.expected_quantiles)
    pred_quantiles = estimator.predict_quantiles(fh=case.fh, alpha=alpha)
    if case.quantile_fh is not None:
        pred_quantiles = _select_steps(case, estimator, pred_quantiles)
    for a in alpha:
        # columns are (variable name, alpha), select by alpha only
        pred_a = pred_quantiles.xs(a, axis=1, level=-1)
        _assert_close(
            case, f"quantile {a} forecast", pred_a, case.expected_quantiles[a]
        )


def _select_steps(case, estimator, pred_quantiles):
    """Select the rows of a quantile forecast at the steps in ``case.quantile_fh``."""
    from sktime.forecasting.base import ForecastingHorizon

    quantile_fh = ForecastingHorizon(case.quantile_fh, freq=estimator.fh.freq)
    index = quantile_fh.to_absolute_index(estimator.cutoff)
    missing = index.difference(pred_quantiles.index)
    if len(missing) > 0:
        raise AssertionError(
            f"golden case {case.name!r}: quantile_fh steps {list(missing)} "
            "are not in fh"
        )
    return pred_quantiles.loc[index]


def _assert_close(case, what, actual, expected):
    """Compare a pandas forecast with a reference array, with a readable message."""
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    # univariate forecasts may come back as a single-column DataFrame
    if expected.ndim == 1 and actual.ndim == 2 and actual.shape[1] == 1:
        actual = actual[:, 0]

    prefix = f"golden case {case.name!r}: {what}"
    if actual.shape != expected.shape:
        raise AssertionError(
            f"{prefix} has shape {actual.shape}, expected {expected.shape}"
        )

    try:
        np.testing.assert_allclose(
            actual, expected, rtol=case.rtol, atol=case.atol, err_msg=prefix
        )
    except AssertionError as e:
        abs_diff = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_diff = abs_diff / np.abs(expected.astype(np.float64))
        msg = (
            f"{prefix} does not match reference "
            f"(rtol={case.rtol}, atol={case.atol}): "
            f"max abs diff {np.nanmax(abs_diff):.6g}, "
            f"max rel diff {np.nanmax(rel_diff):.6g}"
        )
        if case.provenance:
            msg += f"\nreference provenance: {case.provenance}"
        raise AssertionError(f"{msg}\n{e}") from None
