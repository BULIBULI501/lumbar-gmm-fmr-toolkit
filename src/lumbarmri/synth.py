"""Synthetic demo-data generator for the BIC-LASSO pipeline.

Purpose
-------
A reproducible demonstration dataset that exercises the full BIC-LASSO path
(data loading -> standardisation -> grid traversal -> bootstrap band ->
figures) **without any study data**.  The generator plants a known sparse
signal, so the demo can assert that variable selection recovers what was
planted — the standard end-to-end check for a selection pipeline.

Design
------
* 200 samples x 19 columns: ``var_01`` … ``var_18`` predictors + ``y``.
  Generic column names only: nothing that resembles a real study variable.
* Predictors are correlated (AR(1)-like covariance
  ``corr(i, j) = rho^|i-j|``, rho = 0.5) so that selection is non-trivial —
  irrelevant predictors correlate with signal predictors and must be
  penalised away rather than trivially excluded.
* The signal is sparse: exactly 4 of 18 predictors carry a non-zero true
  coefficient (one strong, one medium, one weak, one negative).  ``y`` is a
  linear combination plus Gaussian noise.  The noise standard deviation is
  chosen so the signal is recoverable but not trivially separable.
* Everything is driven by ``numpy.random.default_rng(seed)``: the same seed
  always produces the same table and the same ``synth_truth.json``.

The planted coefficients are written to ``synth_truth.json`` next to the CSV
so any consumer (demo script, tests) can verify recovery without re-deriving
the generator internals.

Usage::

    from lumbarmri.synth import make_clinic, write_demo_data

    df, truth = make_clinic(n_samples=200, seed=42)
    write_demo_data(out_dir)          # writes demo_clinic.csv + synth_truth.json

CLI::

    python -m lumbarmri.synth --output-dir examples/demo_data
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "PLANTED_COEFFICIENTS",
    "make_clinic",
    "write_demo_data",
    "self_check",
]

#: Planted true coefficients (standardised scale): var -> beta.
#: Chosen as strong / medium / weak / negative so the recovery check has a
#: meaningful difficulty gradient.  Exactly these 4 of 18 are non-zero.
PLANTED_COEFFICIENTS: Dict[str, float] = {
    "var_03": 1.20,
    "var_07": -0.80,
    "var_11": 0.55,
    "var_15": -0.35,
}

N_PREDICTORS = 18
RHO = 0.5
NOISE_SD = 1.0
DEFAULT_N_SAMPLES = 200
DEFAULT_SEED = 42


def make_clinic(
    n_samples: int = DEFAULT_N_SAMPLES,
    seed: int = DEFAULT_SEED,
    n_predictors: int = N_PREDICTORS,
    rho: float = RHO,
    noise_sd: float = NOISE_SD,
    coefficients: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Generate the synthetic demo table and its truth record.

    Args:
        n_samples: number of rows.
        seed: RNG seed (fully deterministic).
        n_predictors: number of ``var_NN`` predictors (>= 18 to keep the
            default planted names valid).
        rho: predictor correlation decay, ``corr(i,j) = rho^|i-j|``.
        noise_sd: Gaussian noise standard deviation added to the linear
            signal.
        coefficients: override the planted coefficients (var -> beta).

    Returns:
        ``(df, truth)`` where ``df`` has columns ``var_01..var_NN, y`` and
        ``truth`` is a JSON-serialisable dict describing the planted signal.

    Raises:
        ValueError: n_predictors too small for the planted coefficient names.
    """
    coefficients = dict(coefficients or PLANTED_COEFFICIENTS)
    idx_max = max(int(k.split("_")[1]) for k in coefficients)
    if n_predictors < idx_max:
        raise ValueError(
            f"n_predictors={n_predictors} is too small for planted "
            f"coefficient index {idx_max}"
        )

    rng = np.random.default_rng(seed)
    names = [f"var_{i:02d}" for i in range(1, n_predictors + 1)]

    # AR(1)-like correlation via a Cholesky factor of the Toeplitz covariance.
    idx = np.arange(n_predictors)
    cov = rho ** np.abs(idx[:, None] - idx[None, :])
    chol = np.linalg.cholesky(cov)
    X = rng.standard_normal((n_samples, n_predictors)) @ chol.T

    beta = np.zeros(n_predictors, dtype=float)
    for name, b in coefficients.items():
        beta[names.index(name)] = float(b)

    y = X @ beta + noise_sd * rng.standard_normal(n_samples)

    df = pd.DataFrame(X, columns=names)
    df["y"] = y

    truth: Dict[str, object] = {
        "generator": "lumbarmri.synth.make_clinic",
        "seed": int(seed),
        "n_samples": int(n_samples),
        "n_predictors": int(n_predictors),
        "predictor_correlation": f"AR(1)-like, rho={rho}",
        "noise_sd": float(noise_sd),
        "signal_model": "y = X @ beta + noise_sd * N(0,1); predictors standardised",
        "true_coefficients": {k: float(v) for k, v in coefficients.items()},
        "n_true_signals": len(coefficients),
        "n_noise_predictors": int(n_predictors - len(coefficients)),
        "noise_predictors": [n for n in names if n not in coefficients],
    }
    return df, truth


def write_demo_data(
    output_dir,
    n_samples: int = DEFAULT_N_SAMPLES,
    seed: int = DEFAULT_SEED,
) -> Tuple[Path, Path]:
    """Write ``demo_clinic.csv`` and ``synth_truth.json`` into ``output_dir``.

    Returns:
        ``(csv_path, truth_path)``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df, truth = make_clinic(n_samples=n_samples, seed=seed)

    csv_path = output_dir / "demo_clinic.csv"
    truth_path = output_dir / "synth_truth.json"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    with open(truth_path, "w", encoding="utf-8") as f:
        json.dump(truth, f, ensure_ascii=False, indent=2)
    print(f"wrote {csv_path} ({df.shape[0]} rows x {df.shape[1]} cols)")
    print(f"wrote {truth_path} ({truth['n_true_signals']} planted signals)")
    return csv_path, truth_path


def self_check() -> List[Tuple[str, bool, str]]:
    """Generator self-check: determinism, shape, sparsity, recoverability.

    Returns:
        [(check name, passed, detail)] list.
    """
    results: List[Tuple[str, bool, str]] = []

    def record(name: str, ok: bool, msg: str) -> None:
        results.append((name, bool(ok), msg))

    df1, truth1 = make_clinic(n_samples=200, seed=42)
    df2, truth2 = make_clinic(n_samples=200, seed=42)

    record("same seed reproduces the identical table",
           bool(df1.equals(df2) and truth1 == truth2),
           f"shape={df1.shape}")

    record("table is 200 x 19 (18 predictors + y)",
           df1.shape == (200, 19) and list(df1.columns[-1:]) == ["y"],
           f"shape={df1.shape}, columns ok={list(df1.columns[-1:]) == ['y']}")

    record("column names are generic (var_NN + y)",
           all(c == "y" or (c.startswith("var_") and c[4:].isdigit())
               for c in df1.columns),
           f"columns={list(df1.columns[:3])}...")

    true_vars = set(truth1["true_coefficients"].keys())
    record("exactly 4 planted signals out of 18 predictors",
           len(true_vars) == 4 and truth1["n_noise_predictors"] == 14,
           f"signals={sorted(true_vars)}")

    # Weak recoverability sanity: each planted predictor has a larger
    # |marginal correlation| with y than the median noise predictor.
    corrs = {
        c: abs(float(np.corrcoef(df1[c], df1["y"])[0, 1]))
        for c in df1.columns if c != "y"
    }
    noise_med = float(np.median([corrs[c] for c in corrs if c not in true_vars]))
    weakest = min(true_vars, key=lambda v: corrs[v])
    record("planted signals correlate with y above the noise median",
           corrs[weakest] > noise_med,
           f"weakest signal {weakest} corr={corrs[weakest]:.3f} vs "
           f"noise median={noise_med:.3f}")

    df3, _ = make_clinic(n_samples=200, seed=7)
    record("different seed produces a different table",
           not df1.equals(df3), "seed 7 vs 42")

    return results


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for the synth module CLI."""
    parser = argparse.ArgumentParser(
        description="Generate the synthetic BIC-LASSO demo dataset "
                    "(demo_clinic.csv + synth_truth.json)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  verify
  make --output-dir examples/demo_data [--n-samples 200] [--seed 42]
""",
    )
    parser.add_argument("command", choices=["verify", "make"],
                        help="verify=self-check; make=write demo data files")
    parser.add_argument("--output-dir", type=Path, help="make: target directory")
    parser.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point.

    Returns:
        Exit code: 0 success; 1 self-check failure; 2 argument error.
    """
    args = build_parser().parse_args(argv)

    if args.command == "verify":
        checks = self_check()
        n_fail = len([1 for _, ok, _ in checks if not ok])
        for name, ok, msg in checks:
            print(f"{'[PASS]' if ok else '[FAIL]'} {name}: {msg}")
        print(f"summary: {'all checks passed' if n_fail == 0 else f'{n_fail} check(s) FAILED'}")
        return 0 if n_fail == 0 else 1

    if args.command == "make":
        if args.output_dir is None:
            print("error: --output-dir is required", file=sys.stderr)
            return 2
        write_demo_data(args.output_dir, n_samples=args.n_samples, seed=args.seed)
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
