"""BIC-LASSO variable selection: coefficient path + BIC criterion from one pass.

Why the two figures share one grid traversal
--------------------------------------------
A methods section describing BIC along the full LARS path next to a figure
showing a ten-fold cross-validation curve — while labelling the vertical line
as the BIC optimum — mixes two different tuning procedures in one place.  This
module produces both artefacts from **one pass over a single regular alpha
grid**, so they cannot disagree about which lambda was selected: the same
vertical line appears in both figures by construction.

* coefficient path figure: LASSO coefficients vs log10(alpha) for every
  candidate variable, with the selected lambda marked;
* BIC criterion figure: the criterion along the same grid.

Shared grid
-----------
The coefficient path is not a LARS path: ``Lasso`` is fitted at every point
of a regular grid (default ``np.logspace(-3, 1, 200)``) and the BIC criterion
is evaluated on that same grid.  One traversal therefore yields both the
``(n_alpha, n_var)`` coefficient matrix and the ``(n_alpha,)`` criterion
vector.

Two lambda values, one selected model
-------------------------------------
A direct ``LassoLarsIC`` fit evaluates the criterion only at its (few,
irregular) LARS kinks and reports its own lambda.  Both optimisers retain the
same variable set; the difference is purely where each samples the path.  The
run summary and the legends state both numbers so anyone recomputing with
either algorithm can reconcile them.

Frozen conventions
------------------
* Criterion (recomputed point by point; ``LassoLarsIC.criterion_`` is NOT
  usable because it exists only at the irregular LARS kinks)::

      criterion(a) = n*ln(2*pi*s2) + RSS(a)/s2 + ln(n)*df(a)

  with ``df(a)`` = number of non-zero coefficients at alpha = a
  (``|coef| > nz_tol``) and ``s2`` from one ``LassoLarsIC`` fit
  (sklearn's own noise-variance estimator).
* Predictors are standardized before fitting (LASSO penalises coefficient
  magnitude, so unscaled predictors penalise units, not signal); a guard
  turns the silent failure into an immediate one.
* Bootstrap percentile band with the constant term ``n*ln(2*pi*s2)``
  **retained**, so the band is total uncertainty including the noise-variance
  estimate.  Quote ``lambda_ci`` (not band width) for stability statements.
* Seeds: bootstrap ``RandomState(seed)``; ``Lasso`` itself is deterministic
  on a fixed grid.
* Plot style (journal-safe): pdf/ps fonttype 42 (TrueType), ``svg.fonttype``
  ``none`` (editable text), tab20 palette, green lambda line, red criterion
  curve, turquoise band, all spines visible, linear y axis.

Reproducibility guards
----------------------
* ``assert_standardised`` — rejects unstandardized predictors loudly.
* ``assert_consistency`` — re-runs the grid and requires bit-identical
  coefficient matrix, criterion vector and non-zero counts.
* ``check_against_expected`` — compares against a baseline JSON (written by
  the ``write-baseline`` subcommand) so a data or library change fails loudly
  instead of silently changing the figures.

Configuration
-------------
All study-specific content lives in a YAML file: input CSV, ``y`` column,
candidate variable names, alpha grid, bootstrap parameters, output directory.
This module contains **no real column names** — see
``configs/lasso_template.yaml`` for the annotated blank template.

References
----------
- Efron B, et al. Least angle regression. Ann Stat, 2004, 32(2):407-451.
- Zou H. The adaptive LASSO and its oracle properties. JASA, 2006.
- Tibshirani RJ. Regression shrinkage and selection via the LASSO. JRSS-B,
  1996 (BIC/LASSO model selection along the path).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sklearn.linear_model import Lasso, LassoLarsIC  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import yaml  # noqa: E402

# ---------------------------------------------------------------- style
# Type42 (TrueType) rather than matplotlib's default Type3, which most
# journal preflight checks reject.
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
# Keep SVG text as text. matplotlib defaults to Type3 outlines for SVG, which
# makes labels unsearchable and uneditable in Illustrator.
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["axes.linewidth"] = 1.1
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.top"] = True
plt.rcParams["ytick.right"] = True

# ---------------------------------------------------------------- frozen constants

#: Figure size (double-column journal width ~174 mm at 8x6 in).
FIGSIZE = (8.0, 6.0)
#: Drawn x range of the log10(alpha) axis (matches the default grid).
PLOT_XLIM = (-3.0, 1.0)

MAX_ITER = 50000
NZ_TOL = 1e-8

DEFAULT_N_BOOTSTRAP = 1000
DEFAULT_BOOTSTRAP_SEED = 42
DEFAULT_CI_LEVEL = 0.95
#: Band opacity actually drawn (the published figures rendered 0.20).
DEFAULT_FILL_ALPHA = 0.20

LAM_TOL = 1e-6
CRIT_TOL = 1e-6
NOISE_TOL = 1e-9
LOG10_TOL = 1e-4

TOL_EXACT = 1e-9
TOL_FLOAT = 1e-3

__all__ = [
    "LassoConfig",
    "load_config",
    "validate_columns",
    "load_data",
    "fit_grid",
    "bootstrap_band",
    "assert_standardised",
    "assert_consistency",
    "check_against_expected",
    "check_two_lambdas_agree",
    "run",
    "write_baseline",
    "self_check",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class LassoConfig:
    """BIC-LASSO run configuration (all study-specific content, from YAML).

    Attributes:
        input_csv: source CSV path (read-only).
        output_dir: directory for figures and run_summary.json.
        y_col: outcome column name.
        candidate_vars: candidate predictor column names, in plot order.
        var_name_map: optional ``raw name -> display name`` mapping used only
            for the stdout legend note; plotted labels always use raw names.
        alphas: the regular alpha grid (np.ndarray).
        max_iter: Lasso solver iteration cap.
        nz_tol: ``|coef| > nz_tol`` counts as non-zero.
        n_bootstrap / bootstrap_seed / ci_level / fill_alpha: bootstrap band.
        baseline_path: optional baseline JSON for check_against_expected.
        figsize / xlim: plot geometry.
        config_path: the YAML file this config was loaded from (for the
            summary); None when built programmatically.
    """

    input_csv: Path
    output_dir: Path
    y_col: str
    candidate_vars: List[str]
    alphas: np.ndarray
    var_name_map: Dict[str, str] = field(default_factory=dict)
    max_iter: int = MAX_ITER
    nz_tol: float = NZ_TOL
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED
    ci_level: float = DEFAULT_CI_LEVEL
    fill_alpha: float = DEFAULT_FILL_ALPHA
    baseline_path: Optional[Path] = None
    figsize: Tuple[float, float] = FIGSIZE
    xlim: Tuple[float, float] = PLOT_XLIM
    config_path: Optional[Path] = None

    @property
    def display_names(self) -> Dict[str, str]:
        """Raw -> display mapping with identity fallback."""
        return {v: self.var_name_map.get(v, v) for v in self.candidate_vars}


def _require(cfg: dict, keys: Sequence[str], where: str) -> List[str]:
    """Collect ALL missing keys (report everything at once, never silently)."""
    return [f"{where}.{k}" if where else k for k in keys if k not in cfg]


def load_config(path) -> LassoConfig:
    """Load and validate a YAML configuration.

    Relative paths inside the YAML resolve against the **config file's**
    directory, so a config stays portable.

    Raises:
        FileNotFoundError: config file missing.
        KeyError: required fields missing (all listed in one message).
        ValueError: malformed grid / bootstrap / plot values.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"config root must be a mapping: {path}")

    missing = _require(cfg, ["input_csv", "y_col", "candidate_vars"], "")
    if missing:
        raise KeyError(f"config missing required field(s): {', '.join(missing)}")

    base = path.parent

    def _resolve(p) -> Path:
        p = Path(p)
        return p if p.is_absolute() else (base / p)

    # ---- alpha grid ------------------------------------------------------
    grid = cfg.get("alpha_grid", {"min": 1e-3, "max": 1e1, "num": 200,
                                  "spacing": "log"})
    if "values" in grid:
        alphas = np.asarray(grid["values"], dtype=float)
        if alphas.ndim != 1 or alphas.size < 2:
            raise ValueError("alpha_grid.values must be a 1D array with >= 2 points")
        if np.any(alphas <= 0):
            raise ValueError("alpha_grid.values must be strictly positive")
    else:
        g_missing = _require(grid, ["min", "max", "num"], "alpha_grid")
        if g_missing:
            raise KeyError(f"config missing required field(s): {', '.join(g_missing)}")
        lo, hi, num = float(grid["min"]), float(grid["max"]), int(grid["num"])
        if not (0 < lo < hi):
            raise ValueError(f"alpha_grid needs 0 < min < max, got {lo}, {hi}")
        if num < 2:
            raise ValueError(f"alpha_grid.num must be >= 2, got {num}")
        spacing = str(grid.get("spacing", "log"))
        if spacing == "log":
            alphas = np.logspace(np.log10(lo), np.log10(hi), num)
        elif spacing == "linear":
            alphas = np.linspace(lo, hi, num)
        else:
            raise ValueError(f"alpha_grid.spacing must be 'log' or 'linear', got {spacing}")

    # ---- bootstrap -------------------------------------------------------
    boot = cfg.get("bootstrap", {})
    b_missing = _require(boot, [], "bootstrap")  # all optional, defaults below
    if b_missing:
        raise KeyError(f"config missing required field(s): {', '.join(b_missing)}")

    # ---- plot ------------------------------------------------------------
    plot = cfg.get("plot", {})
    figsize = tuple(plot.get("figsize", FIGSIZE))
    xlim = tuple(plot.get("xlim", PLOT_XLIM))

    vars_ = list(cfg["candidate_vars"])
    if len(vars_) != len(set(vars_)):
        raise ValueError("candidate_vars contains duplicates")
    if cfg["y_col"] in vars_:
        raise ValueError(f"y_col {cfg['y_col']!r} must not be in candidate_vars")

    return LassoConfig(
        input_csv=_resolve(cfg["input_csv"]),
        output_dir=_resolve(cfg.get("output_dir", "outputs/lasso_run")),
        y_col=str(cfg["y_col"]),
        candidate_vars=vars_,
        alphas=alphas,
        var_name_map=dict(cfg.get("var_name_map", {})),
        max_iter=int(cfg.get("max_iter", MAX_ITER)),
        nz_tol=float(cfg.get("nz_tol", NZ_TOL)),
        n_bootstrap=int(boot.get("n_boot", DEFAULT_N_BOOTSTRAP)),
        bootstrap_seed=int(boot.get("seed", DEFAULT_BOOTSTRAP_SEED)),
        ci_level=float(boot.get("ci_level", DEFAULT_CI_LEVEL)),
        fill_alpha=float(boot.get("fill_alpha", DEFAULT_FILL_ALPHA)),
        baseline_path=(_resolve(cfg["baseline_path"])
                       if cfg.get("baseline_path") else None),
        figsize=(float(figsize[0]), float(figsize[1])),
        xlim=(float(xlim[0]), float(xlim[1])),
        config_path=path,
    )


def validate_columns(df: pd.DataFrame, cfg: LassoConfig) -> List[str]:
    """Return every required column missing from ``df`` (empty list = OK).

    The caller prints *all* missing names at once: fixing a config one
    missing column per run is wasted cycles.
    """
    missing = []
    if cfg.y_col not in df.columns:
        missing.append(cfg.y_col)
    missing.extend(v for v in cfg.candidate_vars if v not in df.columns)
    return missing


# ---------------------------------------------------------------------------
# Data + core fitting (frozen algorithm)
# ---------------------------------------------------------------------------


def load_data(cfg: LassoConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(X_raw, y)`` with numeric coercion and complete-case removal.

    CSV is read as strings first (some clinical exports mix formats), every
    required column is coerced numerically, rows with any missing required
    value are dropped.

    Raises:
        KeyError: required columns missing (all listed).
        ValueError: no complete row left.
    """
    df = pd.read_csv(cfg.input_csv, dtype=str)
    missing = validate_columns(df, cfg)
    if missing:
        raise KeyError(
            f"input CSV {cfg.input_csv} is missing required column(s): "
            f"{', '.join(missing)}"
        )
    for c in cfg.candidate_vars + [cfg.y_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df_clean = df[cfg.candidate_vars + [cfg.y_col]].dropna()
    if df_clean.empty:
        raise ValueError(
            f"no complete rows in {cfg.input_csv} over "
            f"{len(cfg.candidate_vars)} candidates + y (all rows had missing values)"
        )
    return (df_clean[cfg.candidate_vars].values.astype(float),
            df_clean[cfg.y_col].values.astype(float))


def fit_grid(X: np.ndarray, y: np.ndarray, alphas: np.ndarray,
             max_iter: int = MAX_ITER, nz_tol: float = NZ_TOL,
             var_names: Optional[Sequence[str]] = None) -> dict:
    """One traversal over ``alphas`` producing both figures' data.

    The BIC criterion cannot be read from ``LassoLarsIC.criterion_``: that
    array is defined only at the LARS kinks, which are irregular and
    decreasing, so it cannot fill a regular grid.  The criterion is therefore
    recomputed point by point:

        criterion(a) = n*ln(2*pi*s2) + RSS(a)/s2 + ln(n)*df(a)

    with ``df(a)`` the number of non-zero coefficients at alpha = a.  The
    noise variance ``s2`` comes from a single ``LassoLarsIC`` fit so the
    estimate uses exactly sklearn's own estimator.  Verified against sklearn:
    at the BIC-optimal lambda the value computed here agrees with
    ``criterion_.min()`` to floating-point noise.

    Returns a dict rather than a positional tuple: the arrays are
    ``(n_alpha, n_var)`` and ``(n_alpha,)`` and an earlier version of this
    code silently compared a coefficient matrix against a criterion vector.
    Names remove the hazard.
    """
    n_obs = X.shape[0]
    ref = LassoLarsIC(criterion="bic", max_iter=int(1e6))
    ref.fit(X, y)
    noise_var = float(ref.noise_variance_)
    const = n_obs * np.log(2 * np.pi * noise_var)

    coef_path = np.empty((len(alphas), X.shape[1]))
    criterion = np.empty(len(alphas))
    n_nonzero = np.empty(len(alphas), dtype=int)

    for i, a in enumerate(alphas):
        m = Lasso(alpha=a, max_iter=max_iter)
        m.fit(X, y)
        coef_path[i] = m.coef_
        resid = y - m.predict(X)
        k = int(np.sum(np.abs(m.coef_) > nz_tol))
        n_nonzero[i] = k
        criterion[i] = const + np.sum(resid ** 2) / noise_var + np.log(n_obs) * k

    names = list(var_names) if var_names is not None else \
        [f"x{j}" for j in range(X.shape[1])]
    return {
        "alphas": alphas,
        "coef_path": coef_path,
        "criterion": criterion,
        "n_nonzero": n_nonzero,
        "noise_var": noise_var,
        "lars_lambda": float(ref.alpha_),
        "lars_selected": [names[j] for j in range(X.shape[1])
                          if np.abs(ref.coef_[j]) > nz_tol],
    }


def bootstrap_band(X: np.ndarray, y: np.ndarray, alphas: np.ndarray,
                   n_boot: int, seed: int, ci_level: float,
                   max_iter: int = MAX_ITER, nz_tol: float = NZ_TOL) -> dict:
    """Percentile bootstrap band for the BIC criterion.

    The constant term ``n*ln(2*pi*s2)`` is **retained**, so the band reflects
    total uncertainty including the noise-variance estimate — which alone can
    shift the whole curve up and down substantially, making the band wide.
    Cite ``lambda_ci`` for stability statements rather than band width.

    The alpha grid is fixed and finite, so every resample yields exactly
    ``len(alphas)`` points and no interpolation is needed.
    """
    n_obs = X.shape[0]
    rng = np.random.RandomState(seed)
    curves = np.empty((n_boot, len(alphas)))
    lambdas = np.empty(n_boot)
    n_zeros = np.empty(n_boot, dtype=int)
    sig2 = np.empty(n_boot)

    for b in range(n_boot):
        idx = rng.choice(n_obs, n_obs, replace=True)
        g = fit_grid(X[idx], y[idx], alphas, max_iter, nz_tol)
        curves[b] = g["criterion"]
        sig2[b] = g["noise_var"]
        lambdas[b] = alphas[int(np.argmin(g["criterion"]))]
        n_zeros[b] = g["n_nonzero"][int(np.argmin(g["criterion"]))]

    lo = np.percentile(curves, 100 * (1 - ci_level) / 2, axis=0)
    hi = np.percentile(curves, 100 * (1 + ci_level) / 2, axis=0)

    pct_lo = 100 * (1 - ci_level) / 2
    pct_hi = 100 * (1 + ci_level) / 2

    return {
        "lo": lo, "hi": hi,
        "lambda_ci": [float(np.percentile(lambdas, pct_lo)),
                      float(np.percentile(lambdas, pct_hi))],
        "lambda_median": float(np.median(lambdas)),
        "n_selected_distribution": {str(k): int(v) for k, v in
                                    zip(*np.unique(n_zeros, return_counts=True))},
        "noise_variance_range": [float(sig2.min()), float(sig2.max())],
    }


# ---------------------------------------------------------------------------
# Reproducibility guards
# ---------------------------------------------------------------------------


def assert_standardised(X: np.ndarray) -> None:
    """Require predictors on a common scale.

    LASSO penalises coefficient magnitude, so unscaled predictors make the
    penalty depend on measurement units.  Omitting the scaler raises nothing:
    it just selects a different lambda.  This guard turns that silent failure
    into an immediate one.

    Raises:
        AssertionError: predictors are not standardized.
    """
    means = np.abs(X.mean(axis=0)).max()
    sds = X.std(axis=0)
    ok = (means < 1e-8) and (sds.min() > 0.9)
    print(f"Standardisation check: max|mean| = {means:.2e}, "
          f"SD range = {sds.min():.4f}~{sds.max():.4f} "
          f"[{'OK' if ok else 'NOT STANDARDISED'}]")
    if not ok:
        raise AssertionError(
            "Predictors are not standardised. LASSO's penalty is applied to "
            "coefficient magnitude, so unscaled predictors would select a "
            "different lambda."
        )


def assert_consistency(X: np.ndarray, y: np.ndarray, alphas: np.ndarray,
                       max_iter: int = MAX_ITER, nz_tol: float = NZ_TOL) -> None:
    """Re-run the grid and require bit-identical output.

    Guards against a library change altering the solution path or the
    non-zero threshold behaviour, either of which would move the selected
    lambda silently.

    Raises:
        AssertionError: the double run disagrees.
    """
    ga = fit_grid(X, y, alphas, max_iter, nz_tol)
    gb = fit_grid(X, y, alphas, max_iter, nz_tol)
    same = (np.array_equal(ga["coef_path"], gb["coef_path"])
            and np.array_equal(ga["criterion"], gb["criterion"])
            and np.array_equal(ga["n_nonzero"], gb["n_nonzero"]))
    print(f"Determinism check: coef_path"
          f"{'' if np.array_equal(ga['coef_path'], gb['coef_path']) else ' DIFF'}, "
          f"criterion"
          f"{'' if np.array_equal(ga['criterion'], gb['criterion']) else ' DIFF'}, "
          f"n_nonzero"
          f"{'' if np.array_equal(ga['n_nonzero'], gb['n_nonzero']) else ' DIFF'} "
          f"[{len(alphas)} alpha points, double run]")
    if not same:
        raise AssertionError(
            "Grid traversal is not deterministic in this environment; "
            "the figures would not be reproducible.")


def check_against_expected(cfg: LassoConfig, n: int, p: int, alphas: np.ndarray,
                           criterion: np.ndarray, n_nonzero: np.ndarray,
                           noise_var: float, lars_lambda: float) -> bool:
    """Compare against the baseline JSON; print a PASS/FAIL table.

    The baseline is produced by the ``write-baseline`` subcommand on the same
    data (for the synthetic demo it is committed to the repository).  Any data
    or library change then fails loudly instead of silently changing figures.
    """
    if cfg.baseline_path is None:
        print("\n--- Consistency with baseline ---")
        print("no baseline_path in config; check skipped")
        return True
    if not Path(cfg.baseline_path).is_file():
        print(f"\n--- Consistency with baseline ---\n"
              f"baseline file not found: {cfg.baseline_path}; check skipped")
        return True
    with open(cfg.baseline_path, "r", encoding="utf-8") as f:
        exp = json.load(f)

    i = int(np.argmin(criterion))
    lam = float(alphas[i])
    nz = int(n_nonzero[i])

    rows = [
        ("Sample count n", exp["n_samples"], n, n == exp["n_samples"]),
        ("Candidate variables", exp["n_candidate_vars"], p,
         p == exp["n_candidate_vars"]),
        ("Alpha grid points", exp["alpha_grid_points"], len(alphas),
         len(alphas) == exp["alpha_grid_points"]),
        ("Noise variance", exp["noise_variance"], noise_var,
         abs(noise_var - exp["noise_variance"]) < NOISE_TOL),
        ("Lambda (grid argmin)", exp["lambda_grid"], lam,
         abs(lam - exp["lambda_grid"]) <= LAM_TOL),
        ("log10(lambda)", exp["log10_lambda"], round(float(np.log10(lam)), 4),
         abs(np.log10(lam) - exp["log10_lambda"]) <= LOG10_TOL),
        ("Selected variables", exp["n_selected"], nz, nz == exp["n_selected"]),
        ("Criterion minimum", exp["criterion_min"], float(criterion.min()),
         abs(criterion.min() - exp["criterion_min"]) <= CRIT_TOL),
        ("Criterion maximum", exp["criterion_max"], float(criterion.max()),
         abs(criterion.max() - exp["criterion_max"]) <= CRIT_TOL),
        ("LARS lambda (legend)", exp["lars_lambda"], lars_lambda,
         abs(lars_lambda - exp["lars_lambda"]) <= LAM_TOL),
    ]

    print("\n--- Consistency with baseline ---")
    print(f"{'Item':<26}{'Expected':<26}{'Current':<24}{'Verdict':<10}")
    print("-" * 86)
    all_ok = True
    for name, e, cur, ok in rows:
        all_ok &= ok
        shown = f"{e:.6g}" if isinstance(e, float) else str(e)
        cur_s = f"{cur:.6g}" if isinstance(cur, float) else str(cur)
        print(f"{name:<26}{shown:<26}{cur_s:<24}{'PASS' if ok else 'FAIL':<10}")
    print("-" * 86)
    print(f"Overall: {'PASS' if all_ok else 'FAIL -- data or library changed'}")
    return all_ok


def check_two_lambdas_agree(selected: Sequence[str], lars_selected: Sequence[str],
                            lam_grid: float, lars_lambda: float) -> bool:
    """The grid optimum and the LARS-kink optimum must retain the same set.

    They differ numerically because each optimiser samples the path at
    different points, but they describe the same model.  This assertion is
    what lets the legends quote both numbers safely.
    """
    same = set(selected) == set(lars_selected)
    print("\n--- Two lambda values, one model ---")
    print(f"{'Grid argmin':<26}{lam_grid:.6f}{'':>10}{len(selected)} variables")
    print(f"{'LARS kink':<26}{lars_lambda:.6f}{'':>10}{len(lars_selected)} variables")
    print("-" * 86)
    print(f"{'Selected set identical':<26}{str(same):<26}{'':<24}"
          f"{'PASS' if same else 'FAIL':<10}")
    if not same:
        print(f"  grid: {list(selected)}")
        print(f"  LARS: {list(lars_selected)}")
    return same


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def plot_fig3_coef_path(ax, log_alphas: np.ndarray, coef_path: np.ndarray,
                        lam_grid: float, n_selected: int,
                        xlim: Tuple[float, float] = PLOT_XLIM) -> None:
    """Coefficient path (style copied from the reference script, not re-chosen).

    tab20 colours cycling over all variables, ``linewidth=1.5``,
    ``alpha=0.85``, a single green vertical line labelled with the BIC
    optimum, bold axis labels at fontsize 14, ``grid(alpha=0.3,
    linestyle='--')``, all four spines visible.  Nothing is added on top: no
    per-variable labels, no legend beyond the reference line, no subtitle.

    A linear y axis is used: in the reference run the maximum absolute
    coefficients span 0.077-1.202 (max-to-median ratio 3.2), so every curve
    stays legible without a log transform.
    """
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(coef_path.shape[1])]

    for i in range(coef_path.shape[1]):
        ax.plot(log_alphas, coef_path[:, i],
                linewidth=1.5, color=colors[i], alpha=0.85)

    ax.axvline(np.log10(lam_grid), color="green", linestyle="-",
               label=f"BIC: log10(lambda)={np.log10(lam_grid):.4f} (nz={n_selected})",
               linewidth=2)

    ax.set_xlabel("log10(lambda)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Standardized Coefficient", fontsize=14, fontweight="bold")
    ax.set_title("LASSO Coefficient Path", fontsize=15, fontweight="bold", pad=15)

    ax.legend(loc="lower right", fontsize=9, frameon=True, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.set_xlim(*xlim)


def plot_fig4_bic_criterion(ax, log_alphas: np.ndarray, criterion: np.ndarray,
                            band: dict, lam_grid: float, n_selected: int,
                            fill_alpha: float = DEFAULT_FILL_ALPHA,
                            xlim: Tuple[float, float] = PLOT_XLIM) -> None:
    """BIC criterion, same style as the coefficient path figure.

    Only the BIC is plotted (no cross-validation curve on a twin axis): two
    different tuning procedures must never share one figure while the line is
    labelled as one of them.  The green line marks the minimum of the curve
    that is actually shown.

    The shaded bootstrap band is drawn underneath with low opacity so it does
    not obscure the curve.
    """
    ax.fill_between(log_alphas, band["lo"], band["hi"], color="turquoise",
                    alpha=fill_alpha, linewidth=0, zorder=1)

    ax.plot(log_alphas, criterion, "-", color="red", linewidth=1.5,
            alpha=0.85, zorder=3)

    ax.axvline(np.log10(lam_grid), color="green", linestyle="-",
               label=f"BIC: log10(lambda)={np.log10(lam_grid):.4f} (nz={n_selected})",
               linewidth=2, zorder=4)

    ax.set_xlabel("log10(lambda)", fontsize=14, fontweight="bold")
    ax.set_ylabel("BIC Criterion", fontsize=14, fontweight="bold")
    ax.set_title("LASSO BIC Criterion Path", fontsize=15, fontweight="bold", pad=15)

    ax.legend(loc="lower right", fontsize=9, frameon=True, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.set_xlim(*xlim)


def save_all(fig, output_dir: Path, stem: str) -> List[Path]:
    """Write PDF, PNG and SVG. SVG keeps text editable."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png", "svg"):
        p = output_dir / f"{stem}.{ext}"
        if ext == "pdf":
            fig.savefig(p, format="pdf", bbox_inches="tight")
        elif ext == "png":
            fig.savefig(p, format="png", dpi=300, bbox_inches="tight")
        else:
            fig.savefig(p, format="svg", bbox_inches="tight")
        paths.append(p)
        print(f"  saved: {p}")
    return paths


# ---------------------------------------------------------------------------
# Run driver
# ---------------------------------------------------------------------------


def run(cfg: LassoConfig, n_bootstrap: Optional[int] = None) -> dict:
    """Full pipeline: load -> standardize -> one grid traversal -> guards ->
    bootstrap band -> two figures -> run_summary.json.

    Returns:
        The run summary dict.
    """
    n_boot = cfg.n_bootstrap if n_bootstrap is None else int(n_bootstrap)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    X_raw, y = load_data(cfg)
    n, p = X_raw.shape
    print(f"  samples = {n}, candidate variables = {p}")
    print(f"  input: {cfg.input_csv}")

    # LASSO penalties are applied to coefficient magnitude, so predictors must
    # be on a common scale or the most strongly unit-valued variable would be
    # penalised hardest purely because of its units.  Skipping this does not
    # raise an error -- it silently selects a different lambda.
    X = StandardScaler().fit_transform(X_raw)
    assert_standardised(X)

    print("\nFitting both figures on one grid traversal ...")
    assert_consistency(X, y, cfg.alphas, cfg.max_iter, cfg.nz_tol)
    grid = fit_grid(X, y, cfg.alphas, cfg.max_iter, cfg.nz_tol, cfg.candidate_vars)
    alphas = grid["alphas"]
    coef_path = grid["coef_path"]
    criterion = grid["criterion"]
    n_nonzero = grid["n_nonzero"]
    noise_var = grid["noise_var"]
    lars_lambda = grid["lars_lambda"]
    lars_sel = grid["lars_selected"]

    i_best = int(np.argmin(criterion))
    lam_grid = float(alphas[i_best])
    selected_idx = [j for j in range(p) if abs(coef_path[i_best, j]) > cfg.nz_tol]
    selected = [cfg.candidate_vars[j] for j in selected_idx]
    n_selected = len(selected_idx)

    print(f"\n  noise variance = {noise_var:.10f}")
    print(f"  criterion range = {criterion.min():.4f} ~ {criterion.max():.4f}")
    print(f"  lambda (grid argmin) = {lam_grid:.6f}"
          f"  (log10 = {np.log10(lam_grid):.4f}), non-zero = {n_selected}")
    print(f"  LARS kink lambda = {lars_lambda:.6f}"
          f"  (log10 = {np.log10(lars_lambda):.4f}), non-zero = {len(lars_sel)}")
    print(f"  retained: {selected}")

    ok = check_against_expected(cfg, n, p, alphas, criterion, n_nonzero,
                                noise_var, lars_lambda)
    ok &= check_two_lambdas_agree(selected, lars_sel, lam_grid, lars_lambda)
    if not ok:
        print("\nWARNING: baseline mismatch -- fix before using these figures.")

    print(f"\nBootstrap band: {n_boot} resamples, seed={cfg.bootstrap_seed}, "
          f"CI={cfg.ci_level:.0%}, fill alpha={cfg.fill_alpha}")
    band = bootstrap_band(X, y, alphas, n_boot, cfg.bootstrap_seed, cfg.ci_level,
                          cfg.max_iter, cfg.nz_tol)
    width = band["hi"] - band["lo"]
    print(f"  band width: median {np.median(width):.1f}, "
          f"max {width.max():.1f}  (criterion span "
          f"{criterion.max() - criterion.min():.1f})")
    print(f"  band range: {band['lo'].min():.1f} ~ {band['hi'].max():.1f}")
    print(f"  lambda {cfg.ci_level:.0%} CI = [{band['lambda_ci'][0]:.6f}, "
          f"{band['lambda_ci'][1]:.6f}]")

    log_alphas = np.log10(alphas)

    print("\nDrawing coefficient-path figure ...")
    fig3, ax3 = plt.subplots(figsize=cfg.figsize)
    plot_fig3_coef_path(ax3, log_alphas, coef_path, lam_grid, n_selected, cfg.xlim)
    p3 = save_all(fig3, cfg.output_dir, "fig3_lasso_coef_path")
    plt.close(fig3)

    print("\nDrawing BIC-criterion figure ...")
    fig4, ax4 = plt.subplots(figsize=cfg.figsize)
    plot_fig4_bic_criterion(ax4, log_alphas, criterion, band, lam_grid, n_selected,
                            cfg.fill_alpha, cfg.xlim)
    p4 = save_all(fig4, cfg.output_dir, "fig4_bic_criterion")
    plt.close(fig4)

    names = cfg.display_names
    summary = {
        "input_csv": str(cfg.input_csv),
        "output_dir": str(cfg.output_dir),
        "y_col": cfg.y_col,
        "n_samples": int(n),
        "n_candidate_vars": int(p),
        "method": (
            "One traversal over a regular alpha grid yields both figures. "
            "criterion(a) = n*ln(2*pi*s2) + RSS(a)/s2 + ln(n)*df(a), with s2 "
            "from LassoLarsIC.noise_variance_. criterion_.min() is not usable "
            "here because it exists only at the irregular LARS kinks."
        ),
        "alpha_grid_points": int(len(alphas)),
        "alpha_grid_range": [float(alphas.min()), float(alphas.max())],
        "noise_variance": float(noise_var),
        "lambda_grid_argmin": lam_grid,
        "log10_lambda_grid_argmin": float(np.log10(lam_grid)),
        "n_selected": n_selected,
        "selected": [
            {"var": cfg.candidate_vars[j], "display_name": names[cfg.candidate_vars[j]],
             "coef": float(coef_path[i_best, j])}
            for j in sorted(selected_idx, key=lambda t: -abs(coef_path[i_best, t]))
        ],
        "lars_kink_reference": {
            "lambda": lars_lambda,
            "log10_lambda": float(np.log10(lars_lambda)),
            "n_selected": len(lars_sel),
            "selected": lars_sel,
            "note": (
                "A direct LassoLarsIC fit evaluates the criterion only at its "
                "LARS kinks and reports its own lambda. The grid value differs "
                "because the grid samples a different set of path points; both "
                "describe the same retained model."
            ),
        },
        "same_selected_set": bool(set(selected) == set(lars_sel)),
        "criterion_min": float(criterion.min()),
        "criterion_max": float(criterion.max()),
        "criterion_span": float(criterion.max() - criterion.min()),
        "argmin_unique_on_grid": bool(
            len(np.where(criterion <= criterion.min() + TOL_EXACT)[0]) == 1),
        "coef_max_abs_range": [float(np.abs(coef_path).max()),
                               float(np.median(np.abs(coef_path).max(axis=0)))],
        "baseline_check_passed": bool(ok),
        "bootstrap": {
            "n_boot": n_boot,
            "seed": cfg.bootstrap_seed,
            "ci_level": cfg.ci_level,
            "fill_alpha": cfg.fill_alpha,
            "constant_term_retained": "n*ln(2*pi*s2) kept, so the band is the "
                                      "total uncertainty",
            "band_range": [float(band["lo"].min()), float(band["hi"].max())],
            "band_width_median": float(np.median(width)),
            "band_width_max": float(width.max()),
            "band_width_over_span": float(np.median(width) /
                                          (criterion.max() - criterion.min())),
            "lambda_ci": band["lambda_ci"],
            "lambda_median": band["lambda_median"],
            "n_selected_distribution": band["n_selected_distribution"],
            "noise_variance_range": band["noise_variance_range"],
        },
        "environment": {
            "python": sys.version.split()[0],
            "sklearn": __import__("sklearn").__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "notes": [
            "Both figures share one grid traversal and therefore one lambda; "
            "the same vertical line appears in both by construction.",
            "The BIC figure contains only the BIC curve: overlaying a "
            "cross-validation curve while labelling the line as the BIC "
            "optimum would put two different tuning procedures in one figure.",
            "Legend labels use the raw CSV column names, so a reader can "
            "match them against the deposited dataset column for column.",
            "The bootstrap band is wide because the constant term "
            "n*ln(2*pi*s2) moves with the noise-variance estimate. Quote "
            "lambda_ci, not the band width, when discussing stability.",
        ],
        "figures": {
            "coef_path": [str(p) for p in p3],
            "bic_criterion": [str(p) for p in p4],
        },
    }
    sp = cfg.output_dir / "run_summary.json"
    with open(sp, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nSummary saved: {sp}")

    print("\n--- Legend note: raw CSV name -> display name ---")
    for j in sorted(selected_idx, key=lambda t: -abs(coef_path[i_best, t])):
        v = cfg.candidate_vars[j]
        print(f"  {v:<20} -> {names[v]:<12} coef = {coef_path[i_best, j]:+.4f}")

    print("\n=== Done ===")
    return summary


def write_baseline(cfg: LassoConfig, output_path) -> Path:
    """Fit one grid traversal and write the baseline JSON.

    The baseline records exactly the values check_against_expected asserts.
    Use it on the data that should stay frozen (the committed synthetic demo
    data, or a private study dataset that must not change silently).
    """
    X_raw, y = load_data(cfg)
    n, p = X_raw.shape
    X = StandardScaler().fit_transform(X_raw)
    assert_standardised(X)
    grid = fit_grid(X, y, cfg.alphas, cfg.max_iter, cfg.nz_tol, cfg.candidate_vars)
    criterion = grid["criterion"]
    i = int(np.argmin(criterion))
    baseline = {
        "generated_by": "lumbarmri.bic_lasso write-baseline",
        "input_csv": str(cfg.input_csv),
        "n_samples": int(n),
        "n_candidate_vars": int(p),
        "alpha_grid_points": int(len(cfg.alphas)),
        "lambda_grid": float(cfg.alphas[i]),
        "log10_lambda": round(float(np.log10(cfg.alphas[i])), 4),
        "n_selected": int(grid["n_nonzero"][i]),
        "noise_variance": float(grid["noise_var"]),
        "criterion_min": float(criterion.min()),
        "criterion_max": float(criterion.max()),
        "lars_lambda": float(grid["lars_lambda"]),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(baseline, f, ensure_ascii=False, indent=2)
    print(f"baseline written: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Self-check (synthetic data; no external dataset needed)
# ---------------------------------------------------------------------------


def self_check() -> List[Tuple[str, bool, str]]:
    """BIC-LASSO self-check on the synthetic demo data.

    Verifies: standardisation guard (raises on unscaled input), grid
    determinism, the frozen BIC formula against a manual recomputation,
    recovery of the planted true signals, bootstrap band sanity, baseline
    round-trip and column validation listing all missing names.

    Returns:
        [(check name, passed, detail)] list.
    """
    results: List[Tuple[str, bool, str]] = []

    def record(name: str, ok: bool, msg: str) -> None:
        results.append((name, bool(ok), msg))

    from .synth import make_clinic

    # Small grid and modest bootstrap keep the self-check fast; the frozen
    # algorithm is identical to the full run.
    alphas = np.logspace(-3, 1, 40)
    df, truth = make_clinic(n_samples=200, seed=42)
    true_vars = set(truth["true_coefficients"].keys())
    var_names = [c for c in df.columns if c != "y"]
    X_raw = df[var_names].values.astype(float)
    y = df["y"].values.astype(float)

    X = StandardScaler().fit_transform(X_raw)
    try:
        assert_standardised(X)
        record("standardisation guard accepts standardized input", True,
               "max|mean| < 1e-8")
    except AssertionError as exc:
        record("standardisation guard accepts standardized input", False, str(exc))

    try:
        assert_standardised(X_raw)
        record("standardisation guard rejects raw input", False,
               "no AssertionError raised")
    except AssertionError:
        record("standardisation guard rejects raw input", True,
               "AssertionError raised as required")

    # ---- determinism -----------------------------------------------------
    ga = fit_grid(X, y, alphas)
    gb = fit_grid(X, y, alphas)
    record("grid traversal is deterministic (double run)",
           bool(np.array_equal(ga["coef_path"], gb["coef_path"])
                and np.array_equal(ga["criterion"], gb["criterion"])
                and np.array_equal(ga["n_nonzero"], gb["n_nonzero"])),
           f"{len(alphas)} alpha points")

    # ---- frozen BIC formula vs manual recomputation ----------------------
    n_obs = X.shape[0]
    i_best = int(np.argmin(ga["criterion"]))
    a_best = alphas[i_best]
    m = Lasso(alpha=a_best, max_iter=MAX_ITER)
    m.fit(X, y)
    resid = y - m.predict(X)
    k = int(np.sum(np.abs(m.coef_) > NZ_TOL))
    manual = (n_obs * np.log(2 * np.pi * ga["noise_var"])
              + np.sum(resid ** 2) / ga["noise_var"] + np.log(n_obs) * k)
    record("criterion = n*ln(2*pi*s2) + RSS/s2 + ln(n)*df (manual recomputation)",
           abs(manual - ga["criterion"][i_best]) < 1e-6,
           f"manual={manual:.6f} vs fit_grid={ga['criterion'][i_best]:.6f}")

    # ---- true-signal recovery -------------------------------------------
    selected = [var_names[j] for j in range(X.shape[1])
                if abs(ga["coef_path"][i_best, j]) > NZ_TOL]
    recovered = true_vars & set(selected)
    record("selected variables recover the planted true signals",
           len(recovered) >= 3,
           f"recovered {sorted(recovered)} of {sorted(true_vars)} "
           f"(selected {len(selected)})")

    # ---- LARS reference ---------------------------------------------------
    record("LARS reference lambda is on the grid range",
           bool(alphas.min() <= ga["lars_lambda"] <= alphas.max()),
           f"lars_lambda={ga['lars_lambda']:.6f}, "
           f"grid=[{alphas.min():.4g}, {alphas.max():.4g}]")

    # ---- bootstrap band ---------------------------------------------------
    band = bootstrap_band(X, y, alphas, n_boot=10, seed=42, ci_level=0.95)
    record("bootstrap band is ordered (lo <= hi)",
           bool(np.all(band["lo"] <= band["hi"] + TOL_FLOAT)),
           f"width median={np.median(band['hi'] - band['lo']):.2f}")
    record("bootstrap lambda_ci is ordered and inside the grid",
           bool(band["lambda_ci"][0] <= band["lambda_ci"][1]
                and alphas.min() <= band["lambda_ci"][0]
                and band["lambda_ci"][1] <= alphas.max()),
           f"lambda_ci={band['lambda_ci']}")

    # ---- baseline round-trip ---------------------------------------------
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        cfg = LassoConfig(
            input_csv=Path(td) / "dummy.csv",  # write_baseline uses its own data below
            output_dir=Path(td),
            y_col="y",
            candidate_vars=var_names,
            alphas=alphas,
        )
        baseline_path = Path(td) / "baseline.json"
        # write_baseline loads from cfg.input_csv; point it at a temp CSV.
        csv_path = Path(td) / "demo.csv"
        df.to_csv(csv_path, index=False)
        cfg.input_csv = csv_path
        write_baseline(cfg, baseline_path)
        cfg.baseline_path = baseline_path
        X_raw2, y2 = load_data(cfg)
        X2 = StandardScaler().fit_transform(X_raw2)
        g2 = fit_grid(X2, y2, alphas)
        ok = check_against_expected(
            cfg, X2.shape[0], X2.shape[1], alphas, g2["criterion"],
            g2["n_nonzero"], g2["noise_var"], g2["lars_lambda"])
        record("baseline round-trip passes check_against_expected", bool(ok),
               f"baseline={baseline_path.name}")

    # ---- column validation -------------------------------------------------
    cfg_bad = LassoConfig(
        input_csv=Path("dummy.csv"),
        output_dir=Path("out"),
        y_col="y_missing",
        candidate_vars=["var_01", "var_missing_a", "var_missing_b"],
        alphas=alphas,
    )
    missing = validate_columns(df, cfg_bad)
    record("validate_columns lists ALL missing columns",
           missing == ["y_missing", "var_missing_a", "var_missing_b"],
           f"missing={missing}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for the bic_lasso module CLI."""
    parser = argparse.ArgumentParser(
        description="BIC-LASSO variable selection: coefficient path + BIC "
                    "criterion from one grid traversal (config-driven)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  verify
  validate --config configs/lasso_demo.yaml
  run          --config configs/lasso_demo.yaml [--n-bootstrap 200]
  write-baseline --config configs/lasso_demo.yaml --output baseline.json
""",
    )
    parser.add_argument("command", choices=["verify", "validate", "run",
                                            "write-baseline"],
                        help="verify=self-check; validate=check config+columns; "
                             "run=full pipeline; write-baseline=freeze baseline JSON")
    parser.add_argument("--config", type=Path, help="YAML configuration file")
    parser.add_argument("--output", type=Path,
                        help="write-baseline: output JSON path")
    parser.add_argument("--n-bootstrap", type=int, default=None,
                        help="run: override bootstrap.n_boot from the config")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point.

    Returns:
        Exit code: 0 success; 1 check/baseline failure; 2 argument or data
        error.
    """
    args = build_parser().parse_args(argv)

    if args.command == "verify":
        checks = self_check()
        n_fail = len([1 for _, ok, _ in checks if not ok])
        for name, ok, msg in checks:
            print(f"{'[PASS]' if ok else '[FAIL]'} {name}: {msg}")
        print(f"summary: {'all checks passed' if n_fail == 0 else f'{n_fail} check(s) FAILED'}")
        return 0 if n_fail == 0 else 1

    if args.config is None:
        print("error: --config is required", file=sys.stderr)
        return 2

    try:
        cfg = load_config(args.config)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.command == "validate":
        print(f"config: {args.config}")
        print(f"  input_csv: {cfg.input_csv} "
              f"[{'found' if cfg.input_csv.is_file() else 'NOT FOUND'}]")
        print(f"  y_col: {cfg.y_col}")
        print(f"  candidate_vars: {len(cfg.candidate_vars)} columns")
        print(f"  alpha grid: {len(cfg.alphas)} points "
              f"[{cfg.alphas.min():.4g}, {cfg.alphas.max():.4g}]")
        print(f"  bootstrap: n_boot={cfg.n_bootstrap}, seed={cfg.bootstrap_seed}, "
              f"ci={cfg.ci_level:.0%}, fill_alpha={cfg.fill_alpha}")
        print(f"  baseline: {cfg.baseline_path or '(none)'}")
        if not cfg.input_csv.is_file():
            print(f"FAIL: input CSV not found: {cfg.input_csv}")
            return 2
        df = pd.read_csv(cfg.input_csv, dtype=str)
        missing = validate_columns(df, cfg)
        if missing:
            print(f"FAIL: input CSV is missing {len(missing)} required column(s):")
            for c in missing:
                print(f"  - {c}")
            return 2
        print(f"OK: all {1 + len(cfg.candidate_vars)} required columns present")
        return 0

    if args.command == "run":
        try:
            summary = run(cfg, n_bootstrap=args.n_bootstrap)
        except (KeyError, ValueError, AssertionError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1 if isinstance(exc, AssertionError) else 2
        return 0 if summary.get("baseline_check_passed", True) else 1

    if args.command == "write-baseline":
        if args.output is None:
            print("error: --output is required for write-baseline", file=sys.stderr)
            return 2
        try:
            write_baseline(cfg, args.output)
        except (KeyError, ValueError, AssertionError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1 if isinstance(exc, AssertionError) else 2
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
