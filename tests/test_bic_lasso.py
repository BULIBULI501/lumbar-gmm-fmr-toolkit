"""Golden-value tests for the BIC-LASSO core (small, fast — no full grid)."""

import numpy as np
import pytest
from sklearn.linear_model import Lasso, LassoLarsIC
from sklearn.preprocessing import StandardScaler

from lumbarmri.bic_lasso import (
    NZ_TOL,
    assert_standardised,
    check_two_lambdas_agree,
    fit_grid,
)
from lumbarmri.synth import PLANTED_COEFFICIENTS, make_clinic


def test_bic_criterion_independent_recompute():
    """The criterion must equal n*ln(2*pi*s2) + RSS/s2 + ln(n)*df, recomputed
    here from sklearn primitives rather than trusting the module output."""
    rng = np.random.RandomState(0)
    X = StandardScaler().fit_transform(rng.normal(size=(20, 3)))
    y = X @ np.array([1.0, -0.5, 0.0]) + rng.normal(scale=0.3, size=20)

    alpha = 0.05
    g = fit_grid(X, y, np.array([alpha]), var_names=["a", "b", "c"])

    m = Lasso(alpha=alpha).fit(X, y)
    resid = y - m.predict(X)
    k = int(np.sum(np.abs(m.coef_) > NZ_TOL))
    ref = LassoLarsIC(criterion="bic", max_iter=int(1e6)).fit(X, y)
    s2 = float(ref.noise_variance_)
    n = X.shape[0]

    expected = n * np.log(2 * np.pi * s2) + float(np.sum(resid ** 2)) / s2 + np.log(n) * k
    assert abs(float(g["criterion"][0]) - expected) < 1e-9
    assert abs(float(g["noise_var"]) - s2) < 1e-12
    assert int(g["n_nonzero"][0]) == k


def test_bic_prefers_truthful_model():
    """At the grid optimum the BIC must prefer the planted model over noise."""
    rng = np.random.RandomState(1)
    X = StandardScaler().fit_transform(rng.normal(size=(40, 4)))
    beta = np.array([1.2, 0.0, -0.8, 0.0])
    y = X @ beta + rng.normal(scale=0.5, size=40)

    alphas = np.logspace(-3, 1, 30)
    g = fit_grid(X, y, alphas, var_names=["v1", "v2", "v3", "v4"])
    best = int(np.argmin(g["criterion"]))
    selected = {name for name, c in zip(["v1", "v2", "v3", "v4"], g["coef_path"][best]) if abs(c) > NZ_TOL}
    assert {"v1", "v3"} <= selected
    assert "v2" not in selected and "v4" not in selected


def test_synth_signal_recovery():
    """On synthetic clinic data the two strongest planted signals are found."""
    df, truth = make_clinic(n_samples=120, seed=42)
    assert truth["true_coefficients"] == PLANTED_COEFFICIENTS

    names = [c for c in df.columns if c != "y"]
    X = StandardScaler().fit_transform(df[names].values.astype(float))
    y = df["y"].values.astype(float)

    alphas = np.logspace(-3, 1, 30)
    g = fit_grid(X, y, alphas, var_names=names)
    best = int(np.argmin(g["criterion"]))
    selected = {name for name, c in zip(names, g["coef_path"][best]) if abs(c) > NZ_TOL}
    assert "var_03" in selected, f"strongest planted signal missed: {sorted(selected)}"
    assert "var_07" in selected, f"second planted signal missed: {sorted(selected)}"


def test_assert_standardised_rejects_raw_scale():
    X = np.array([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0]])
    with pytest.raises(AssertionError):
        assert_standardised(X)


def test_assert_standardised_accepts_z_scores():
    rng = np.random.RandomState(2)
    X = StandardScaler().fit_transform(rng.normal(size=(50, 3)))
    assert_standardised(X)  # must not raise


def test_two_lambdas_agree_logic():
    assert check_two_lambdas_agree(["a", "b"], ["b", "a"], 0.1, 0.09) is True
    assert check_two_lambdas_agree(["a", "b"], ["a", "c"], 0.1, 0.09) is False
