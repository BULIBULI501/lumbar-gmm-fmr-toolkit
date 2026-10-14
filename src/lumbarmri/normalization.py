"""Intensity normalization and harmonization for lumbar paraspinal MRI.

T1/T2-weighted MRI is not quantitative: the same fat tissue can differ by
several times in grayscale across scanners, time points and reconstruction
settings, so raw intensities are not comparable across cases.

Order matters — these are two different jobs, never reversed:

* **Bias-field correction** (mod :mod:`lumbarmri.preprocess`) removes the
  spatially varying multiplicative field *inside one image*.
* **Intensity normalization** (this module) aligns the grayscale *scale
  between images* (affine / histogram).

Normalization statistics depend on absolute grayscale, so N4 must run first.

Four routes, by recommendation:

============================  =======================  ================================
method                        fits                     one-liner
============================  =======================  ================================
:func:`fit_nyul`/:func:`      multi-parameter MRI,     11 histogram landmarks mapped
apply_nyul` (Nyul)            recommended              to a trained standard scale
:func:`histogram_match`       same protocol/region     quantile matching to a reference
:func:`zscore` (ROI)          small samples,           mean 0 / std 1 inside the ROI,
                              scale removal only       anatomy contrast lost
:func:`combat_harmonize`      multi-center/multi-      batch-effect removal on the
(ComBat, optional extra)      scanner                  *feature table*, not the image
============================  =======================  ================================

ComBat works at a different level: it removes batch mean/variance shifts from
a feature matrix and cannot replace image-level N4 / normalization.

Effect on GMM segmentation: :func:`zscore` and :func:`apply_nyul` are monotone
transforms, so they do **not** change voxel cluster assignment for
intensity-clustering downstream steps (GMM fat = highest-mean cluster).  The
value of monotonicity is cross-device comparability of absolute-intensity
features.

Statistics convention: ROI grayscale statistics use **ddof=0** (population
std), matching the ICC/FMR baselines of the source study.

References
----------
- Nyul LG, et al. IEEE TMI, 2000, 19(2):143-150.
- Shah M, et al. Med Image Anal, 2011, 15(2):267-282.
- Johnson WE, et al. Biostatistics, 2007, 8(1):118-127. (ComBat)
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import SimpleITK as sitk

LOGGER = logging.getLogger(__name__)

ArrayOrImage = Union[np.ndarray, sitk.Image]

TOL_EXACT = 1e-9
TOL_FLOAT = 1e-3

__all__ = [
    "roi_stats",
    "zscore",
    "histogram_match",
    "NyulModel",
    "fit_nyul",
    "apply_nyul",
    "combat_harmonize",
    "self_check",
]


def _as_array(x: ArrayOrImage) -> np.ndarray:
    return x if isinstance(x, np.ndarray) else sitk.GetArrayFromImage(x)


def _as_mask(mask: Optional[sitk.Image], ref: sitk.Image) -> sitk.Image:
    if mask is None:
        m = sitk.GetImageFromArray(
            np.ones(sitk.GetArrayViewFromImage(ref).shape, dtype=np.uint8)
        )
        m.CopyInformation(ref)
        return sitk.Cast(m, sitk.sitkUInt8)
    m = sitk.Cast(mask > 0, sitk.sitkUInt8)
    m.CopyInformation(ref)
    return m


def roi_stats(
    image: ArrayOrImage, mask: Optional[sitk.Image] = None
) -> Tuple[float, float]:
    """ROI grayscale mean and standard deviation (**ddof=0**, population std).

    Deliberately not using SimpleITK statistics operators — three measured
    limitations: ``LabelStatisticsImageFilter`` is unavailable in 3D;
    ``StatisticsImageFilter`` has no ``SetMask``; and ``sitk.Mask(img, mask,
    0)`` followed by statistics counts the outside fill values in the
    denominator (measured: a true ROI mean of 547.03 came out as 136.76;
    swapping the sentinel value does not help — the sentinel itself is a data
    point).  ``GetArrayViewFromImage`` returns a read-only zero-copy view used
    for scalar statistics only; every pixel-level transform still runs on
    SimpleITK operators.

    Args:
        image: SimpleITK image or ndarray (z, y, x).
        mask: ROI mask; ``None`` = whole image.

    Returns:
        ``(mean, std)``; ``(nan, nan)`` for an empty ROI.
    """
    arr = _as_array(image)
    vals = arr.reshape(-1) if mask is None else arr[_as_array(mask) > 0]
    if vals.size == 0:
        return float("nan"), float("nan")
    return float(vals.mean()), float(vals.std(ddof=0))


def zscore(
    image: ArrayOrImage,
    mask: Optional[sitk.Image] = None,
    clip_sigma: Optional[float] = None,
) -> sitk.Image:
    """ROI z-score normalization (mean 0, variance 1) on SimpleITK operators.

    Negative z-scores *inside* the ROI are expected (voxels below the ROI
    mean); the invariant to hold is **zero outside the mask**, else background
    leaks into downstream features.

    Degenerate ROI (no variance or empty): z-score is undefined, so the
    original grayscale is kept and only the outside is zeroed — distinct from
    returning all zeros ("normalization failed").

    ``clip_sigma`` applies symmetric truncation via
    :class:`~SimpleITK.ClampImageFilter`, *not* ``IntensityWindowing`` — the
    latter is a window/level display mapping (window range mapped to the
    output pixel range), not a clamp.  Measured with 2.5.3: windowing z-scores
    of (-0.52..2.60) with a (-3, 3) window produced 105..238.

    Note:
        ``ShiftScaleImageFilter`` must set the output pixel type explicitly:
        by default it *inherits the input type*, and an int16 input truncates
        z-scores to integers (measured: ROI std 0.65 instead of 1.0, 244 gray
        levels collapsed to 3).
    """
    img = image if isinstance(image, sitk.Image) else sitk.GetImageFromArray(
        np.asarray(image, dtype=np.float32)
    )
    m = _as_mask(mask, img)
    mu, sd = roi_stats(img, m)
    if not np.isfinite(sd) or sd == 0:
        LOGGER.warning(
            "ROI grayscale not standardizable (mean=%s, sd=%s); keeping raw gray", mu, sd
        )
        return sitk.Cast(sitk.Mask(img, m, outsideValue=0.0), sitk.sitkFloat32)

    shift_scale = sitk.ShiftScaleImageFilter()
    shift_scale.SetShift(-mu)
    shift_scale.SetScale(1.0 / sd)
    shift_scale.SetOutputPixelType(sitk.sitkFloat32)
    out = sitk.Mask(shift_scale.Execute(img), m, outsideValue=0.0)

    if clip_sigma is not None:
        clamp = sitk.ClampImageFilter()
        clamp.SetLowerBound(float(-clip_sigma))
        clamp.SetUpperBound(float(clip_sigma))
        clamp.SetOutputPixelType(sitk.sitkFloat32)
        out = clamp.Execute(out)

    return sitk.Cast(out, sitk.sitkFloat32)


def histogram_match(
    image: sitk.Image,
    reference: sitk.Image,
    levels: int = 64,
    match_points: int = 16,
    threshold_at_mean: bool = True,
) -> sitk.Image:
    """Histogram matching: align the source grayscale distribution to a reference.

    Fits same-protocol, same-region images registered to one reference.
    ``threshold_at_mean`` excludes background voxels below the mean before
    histogram statistics — the ITK documentation explicitly notes the
    algorithm performs worst when background is not excluded (especially for
    small-FOV body images).

    Note:
        ``HistogramMatchingImageFilter`` only offers ``Execute(image,
        reference)`` — there is no ``GetOutput()``.
    """
    f = sitk.HistogramMatchingImageFilter()
    f.SetNumberOfHistogramLevels(int(levels))
    f.SetNumberOfMatchPoints(int(match_points))
    f.ThresholdAtMeanIntensityOn() if threshold_at_mean else f.ThresholdAtMeanIntensityOff()
    return f.Execute(
        sitk.Cast(image, sitk.sitkFloat32), sitk.Cast(reference, sitk.sitkFloat32)
    )


@dataclass
class NyulModel:
    """Nyul standard scale (histogram landmarks).

    ``landmarks`` has length ``num_points + 2``:
    ``[p_low, m1, ..., m_num_points, p_high]`` where ``p_low`` / ``p_high``
    come from the **whole-image** CDF (locking the range) and the interior
    landmarks from the **foreground** histogram (describing tissue
    distribution).
    """

    landmarks: np.ndarray
    num_points: int = 10
    p_low: float = 1.0
    p_high: float = 99.0
    train_landmarks: Optional[np.ndarray] = None

    def save(self, path) -> None:
        """Persist as ``.npz`` (``allow_pickle=False`` compatible)."""
        np.savez(
            str(path),
            landmarks=self.landmarks,
            num_points=self.num_points,
            p_low=self.p_low,
            p_high=self.p_high,
            train_landmarks=(
                self.train_landmarks
                if self.train_landmarks is not None
                else np.empty((0, len(self.landmarks)))
            ),
        )

    @classmethod
    def load(cls, path) -> "NyulModel":
        """Load from ``.npz`` written by :meth:`save`."""
        with np.load(str(path), allow_pickle=False) as z:
            return cls(
                landmarks=z["landmarks"],
                num_points=int(z["num_points"]),
                p_low=float(z["p_low"]),
                p_high=float(z["p_high"]),
                train_landmarks=z["train_landmarks"],
            )


def _landmarks_of(
    arr: np.ndarray, mask: Optional[np.ndarray], num_points: int, p_low: float, p_high: float
) -> np.ndarray:
    """Landmarks of one image: ``[p_low, m1, ..., m_num_points, p_high]``.

    ``p_low`` / ``p_high`` come from the **whole-image** CDF to lock the
    range; the interior landmarks come from the **foreground** histogram to
    describe tissue distribution.  Length is always ``num_points + 2``.

    Interior points use ``arange``-spaced percentiles
    ``i * 100 / (num_points + 1)``, i = 1..num_points.  The linspace slice
    ``linspace(0, 100, num_points + 1)[1:-1]`` would yield only
    ``num_points - 1`` interior points (9 for num_points=10), silently
    shrinking the model.
    """
    deciles = np.arange(1, num_points + 1) * (100.0 / (num_points + 1))
    fg = arr if mask is None else arr[mask]
    if fg.size == 0:
        raise ValueError("empty foreground ROI, cannot compute Nyul landmarks")
    values = [float(np.percentile(arr, p_low))]
    values += [float(np.percentile(fg, p)) for p in deciles]
    values.append(float(np.percentile(arr, p_high)))
    return np.asarray(values, dtype=np.float64)


def fit_nyul(
    samples: Sequence[Tuple[np.ndarray, Optional[np.ndarray]]],
    num_points: int = 10,
    p_low: float = 1.0,
    p_high: float = 99.0,
) -> NyulModel:
    """Learn the Nyul standard scale from a training set.

    Args:
        samples: ``[(image array, foreground mask array or None), ...]``.
        num_points: number of interior foreground landmarks.
        p_low: whole-image lower percentile locking the range.
        p_high: whole-image upper percentile locking the range.

    Returns:
        NyulModel whose ``landmarks`` is the per-sample landmark mean;
        ``train_landmarks`` keeps every sample's landmarks for QC plots.
    """
    if not samples:
        raise ValueError("empty training set")
    lms = np.vstack([
        _landmarks_of(_as_array(img), mask, num_points, p_low, p_high)
        for img, mask in samples
    ])
    return NyulModel(
        landmarks=lms.mean(axis=0),
        num_points=num_points,
        p_low=p_low,
        p_high=p_high,
        train_landmarks=lms,
    )


def apply_nyul(
    image: sitk.Image,
    model: NyulModel,
    mask: Optional[sitk.Image] = None,
) -> sitk.Image:
    """Apply the Nyul standard scale to one image.

    Piecewise-linear mapping with linear extrapolation at both ends; strictly
    monotone by construction, so grayscale order is preserved — the required
    contract for downstream "fat = highest-intensity cluster" segmentation.
    SimpleITK has no LUT operator (``RescaleIntensityImageFilter`` exposes no
    table interface), hence numpy.

    Raises:
        ValueError: fewer than 2 distinct landmarks after deduplication.
    """
    arr = _as_array(image)
    m = None if mask is None else (_as_array(mask) > 0)
    src = _landmarks_of(arr, m, model.num_points, model.p_low, model.p_high)
    ref = np.asarray(model.landmarks, dtype=np.float64)

    keep = np.ones(src.size, dtype=bool)
    keep[1:-1] = np.diff(src[:-1]) > 0  # dedupe for strict monotonicity
    if keep.sum() < 2:
        raise ValueError(f"fewer than 2 valid landmarks after dedup: {src}")
    x, y = src[keep], ref[keep]

    out = np.interp(arr, x, y).astype(np.float32)
    # Linear extrapolation at both ends: without it, grayscale outside the
    # landmark range collapses to a constant.
    lo, hi = arr < x[0], arr > x[-1]
    if lo.any() and x[1] > x[0]:
        k = (y[1] - y[0]) / (x[1] - x[0])
        out[lo] = y[0] + (arr[lo] - x[0]) * k
    if hi.any() and x[-1] > x[-2]:
        k = (y[-1] - y[-2]) / (x[-1] - x[-2])
        out[hi] = y[-1] + (arr[hi] - x[-1]) * k

    result = sitk.GetImageFromArray(out)
    result.CopyInformation(image)
    return result


# ---------------------------------------------------------------------------
# ComBat harmonization (feature level; optional extra)
# ---------------------------------------------------------------------------


def combat_harmonize(
    features,
    batch_col: str,
    covariate_cols: Optional[Sequence[str]] = None,
    categorical_cols: Optional[Sequence[str]] = None,
):
    """ComBat batch-effect harmonization (requires ``pip install neuroCombat``).

    Level of action: removes batch mean/variance shifts from the **feature
    matrix** while preserving the biological effects of the given covariates.
    It cannot replace image-level N4 / intensity normalization.

    Args:
        features: samples x features table (pandas DataFrame) including
            ``batch_col`` and the covariate columns.
        batch_col: batch / scanner / site identifier column.
        covariate_cols: covariates whose biology must survive (continuous
            such as age; categorical such as sex or disease status).
        categorical_cols: columns declared categorical (remaining covariates
            are treated as continuous).

    Returns:
        Copy of ``features`` with feature columns replaced by the harmonized
        values.
    """
    import pandas as pd  # local import: pandas only needed for this function

    try:
        from neuroCombat import neuroCombat  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "ComBat requires the neuroCombat package: pip install neuroCombat "
            "(or: pip install lumbarmri[combat])"
        ) from exc

    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame")
    if batch_col not in features.columns:
        raise KeyError(f"batch column not found: {batch_col}")
    covariate_cols = list(covariate_cols or [])
    categorical_cols = list(categorical_cols or [])
    covars = features[[batch_col] + covariate_cols].copy()
    feat_cols = [c for c in features.columns if c not in covars.columns]
    covars[batch_col] = covars[batch_col].astype(str)

    # neuroCombat requires float data without missing values; non-numeric
    # feature columns are rejected explicitly.
    dat = features[feat_cols].apply(pd.to_numeric, errors="coerce")
    if dat.isna().any().any():
        bad = dat.columns[dat.isna().any()].tolist()
        raise ValueError(f"feature columns with non-numeric or missing values: {bad[:5]}")
    if len(np.unique(covars[batch_col])) < 2:
        raise ValueError(
            "batch column has a single unique value; ComBat needs >= 2 batches"
        )

    out = neuroCombat(
        dat=dat,
        covars=covars,
        batch_col=batch_col,
        categorical_cols=[c for c in categorical_cols if c in covars.columns],
        continuous_cols=[c for c in covariate_cols if c not in categorical_cols],
    )["data"]
    result = features.copy()
    result[feat_cols] = out
    return result


# ---------------------------------------------------------------------------
# Self-check (synthetic data; no external dataset needed)
# ---------------------------------------------------------------------------


def self_check() -> List[Tuple[str, bool, str]]:
    """Normalization self-check (anchors C/D of the research verify command).

    C. z-score invariants: ROI mean ~ 0 / std ~ 1 (1e-3), outside-mask exactly
       0, ``clip_sigma`` truncation actually clamps (deterministic outlier
       probe), degenerate ROI keeps raw gray.
    D. Operator/reference equivalence of z-score (1e-4), ddof=0 statistics
       convention (1e-3), histogram matching finite output, Nyul landmarks
       strictly increasing + apply finite + ``.npz`` round-trip, ComBat
       dependency guard.

    Returns:
        [(check name, passed, detail)] list.
    """
    results: List[Tuple[str, bool, str]] = []

    def record(name: str, ok: bool, msg: str) -> None:
        results.append((name, bool(ok), msg))

    # Two-tissue phantom corrected by N4 3D (same input as the research
    # verify section C).  Function-level import avoids the circular module
    # dependency: preprocess imports roi_stats from this module.
    from .preprocess import N4Params, correct_3d, synthetic_phantom

    img, mask, _ = synthetic_phantom(two_tissue=True, seed=42)
    corr = correct_3d(img, mask, N4Params()).corrected
    arr = _as_array(corr)
    m = _as_array(mask) > 0

    # ---- C1. z-score invariants -----------------------------------------
    zs = zscore(corr, mask)
    zv = _as_array(zs)[m]
    record("C1: z-score ROI mean ~ 0 (1e-3)", abs(zv.mean()) < 1e-3,
           f"mean={zv.mean():.2e}")
    record("C1: z-score ROI std ~ 1 (1e-3)", abs(zv.std() - 1) < 1e-3,
           f"std={zv.std():.4f}")
    out_abs = float(np.abs(_as_array(zs)[~m]).max())
    record("C1: z-score is 0 outside the mask (no background leak)",
           out_abs <= TOL_EXACT, f"max|outside|={out_abs:.2e}")

    # ---- C2. clip semantics ---------------------------------------------
    zs_clip = zscore(corr, mask, clip_sigma=3.0)
    zc = _as_array(zs_clip)[m]
    record("C2: clip_sigma=3 bounds hold (1e-6)",
           zc.max() <= 3.0 + 1e-6 and zc.min() >= -3.0 - 1e-6,
           f"range=[{zc.min():.4f}, {zc.max():.4f}]")
    out_abs = float(np.abs(_as_array(zs_clip)[~m]).max())
    record("C2: clipped z-score still 0 outside the mask",
           out_abs <= TOL_EXACT, f"max|outside|={out_abs:.2e}")
    # Deterministic clamp probe: one huge outlier forces z >> 3, so the
    # clipped maximum must land exactly on +3 (a real clamp, not windowing —
    # windowing would remap to the output pixel range).
    probe = np.zeros((4, 16, 16), dtype=np.float32)
    probe_mask = np.ones_like(probe, dtype=np.uint8)
    probe[2, 8, 8] = 1000.0
    probe_img = sitk.GetImageFromArray(probe)
    probe_mask_img = sitk.Cast(sitk.GetImageFromArray(probe_mask), sitk.sitkUInt8)
    zs_probe = zscore(probe_img, probe_mask_img, clip_sigma=3.0)
    vmax = float(_as_array(zs_probe).max())
    record("C2: clamp truncates large z exactly to +3 (1e-3)",
           abs(vmax - 3.0) < 1e-3, f"max={vmax:.4f}")

    # ---- C3. degenerate ROI keeps raw gray ------------------------------
    flat = sitk.GetImageFromArray(np.full((4, 8, 8), 500.0, dtype=np.float32))
    flat_mask = sitk.Cast(sitk.GetImageFromArray(np.ones((4, 8, 8), np.uint8)),
                          sitk.sitkUInt8)
    zs_flat = _as_array(zscore(flat, flat_mask))
    record("C3: constant ROI keeps raw grayscale (std=0 path)",
           abs(zs_flat.mean() - 500.0) < TOL_FLOAT,
           f"mean={zs_flat.mean():.4f}")

    # ---- D1. operator vs numpy reference equivalence ---------------------
    ref_z = (arr - arr[m].mean()) / arr[m].std()
    ref_z = np.where(m, ref_z, 0.0).astype(np.float32)
    diff = float(np.abs(_as_array(zs) - ref_z).max())
    record("D1: operator z-score matches numpy reference (1e-4)",
           diff <= 1e-4, f"max_diff={diff:.2e}")

    # ---- D2. ddof=0 statistics convention --------------------------------
    mu_s, sd_s = roi_stats(corr, mask)
    ref_vals = arr[m]
    record("D2: roi_stats std is ddof=0 (1e-3)",
           abs(sd_s - ref_vals.std(ddof=0)) <= 1e-3,
           f"roi_stats={sd_s:.6f} vs numpy={ref_vals.std(ddof=0):.6f}")
    record("D2: roi_stats mean matches numpy (1e-3)",
           abs(mu_s - ref_vals.mean()) <= 1e-3, f"mean={mu_s:.6f}")

    # ---- D3. histogram matching ------------------------------------------
    ref = sitk.GetImageFromArray((arr * 1.7 + 30.0).astype(np.float32))
    ref.CopyInformation(corr)
    hm = _as_array(histogram_match(corr, ref, threshold_at_mean=False))
    record("D3: histogram matching output is finite",
           bool(np.all(np.isfinite(hm))), f"range=[{hm.min():.1f}, {hm.max():.1f}]")

    # ---- D4. Nyul train/apply/npz round-trip ------------------------------
    model = fit_nyul([(_as_array(img), None), (arr, None)])
    lm0 = model.train_landmarks[0]
    record("D4: Nyul per-sample landmarks strictly increasing",
           bool(np.all(np.diff(lm0) > 0)), f"landmarks={np.round(lm0, 1)}")
    norm = _as_array(apply_nyul(corr, model))
    record("D4: Nyul apply output is finite", bool(np.all(np.isfinite(norm))),
           f"range=[{norm.min():.1f}, {norm.max():.1f}]")
    # Monotone map: grayscale order must be preserved (GMM fat = max mean).
    order_src = np.argsort(arr, axis=None)[:200]
    order_ok = bool(
        np.all(np.diff(norm.ravel()[order_src]) >= -TOL_FLOAT)
    )
    record("D4: Nyul map preserves grayscale order",
           order_ok, "sorted-input output non-decreasing")

    with tempfile.TemporaryDirectory() as td:
        npz = Path(td) / "nyul_model.npz"
        model.save(npz)
        model2 = NyulModel.load(npz)
        same = (
            np.allclose(model.landmarks, model2.landmarks, atol=TOL_EXACT)
            and model.num_points == model2.num_points
            and abs(model.p_low - model2.p_low) < TOL_EXACT
            and abs(model.p_high - model2.p_high) < TOL_EXACT
        )
        record("D4: NyulModel .npz round-trip preserves the scale",
               bool(same), f"landmarks equal={np.allclose(model.landmarks, model2.landmarks)}")

    # ---- D5. ComBat dependency guard -------------------------------------
    import pandas as pd

    df = pd.DataFrame(
        {
            "batch": ["A"] * 6 + ["B"] * 6,
            "age": [60.0] * 12,
            "feat_1": np.r_[np.full(6, 100.0), np.full(6, 130.0)],
            "feat_2": np.r_[np.full(6, 50.0), np.full(6, 55.0)],
        }
    )
    try:
        import neuroCombat  # noqa: F401

        out = combat_harmonize(df, "batch", covariate_cols=["age"])
        record("D5: ComBat harmonization runs on a synthetic 2-batch table",
               bool(np.all(np.isfinite(out[["feat_1", "feat_2"]].to_numpy()))),
               f"output shape={out.shape}")
    except ImportError:
        try:
            combat_harmonize(df, "batch", covariate_cols=["age"])
            record("D5: ComBat missing dependency raises ImportError", False,
                   "call succeeded although neuroCombat is not installed")
        except ImportError:
            record("D5: ComBat missing dependency raises ImportError", True,
                   "neuroCombat not installed (optional extra), guard works")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser():
    """Build the argparse parser for the normalization module CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Intensity normalization (z-score / histogram matching / Nyul) "
                    "and ComBat feature harmonization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  verify
  zscore     --image x_n4.nii.gz --mask x_ME_label.nii.gz --output x_n4_zs.nii.gz
  hist-match --image x_n4.nii.gz --reference ref_n4.nii.gz --output x_hm.nii.gz
  nyul-train --root ./cases --model-out nyul_model.npz
  nyul-apply --root ./cases --model nyul_model.npz --output-dir out
  combat     --features feat.csv --batch-col scanner --covariate-cols age,sex \\
             --output feat_combat.csv
""",
    )
    parser.add_argument(
        "command",
        choices=["verify", "zscore", "hist-match", "nyul-train", "nyul-apply", "combat"],
        help="verify=self-check; zscore/hist-match=single image; "
             "nyul-train/nyul-apply=case tree; combat=feature table",
    )
    parser.add_argument("--image", type=Path, help="zscore/hist-match: source image")
    parser.add_argument("--reference", type=Path, help="hist-match: reference image")
    parser.add_argument("--mask", type=Path, help="zscore: ROI mask")
    parser.add_argument("--output", type=Path,
                        help="zscore/hist-match/combat: output path")
    parser.add_argument("--clip-sigma", type=float, default=None,
                        help="zscore: symmetric z truncation, e.g. 3.0")
    parser.add_argument("--root", type=Path, help="nyul-train/nyul-apply: data root")
    parser.add_argument("--case-list", type=Path,
                        help="nyul-train/nyul-apply: text file of case ids")
    parser.add_argument("--model-out", type=Path, help="nyul-train: output .npz")
    parser.add_argument("--model", type=Path, help="nyul-apply: trained .npz")
    parser.add_argument("--output-dir", type=Path, help="nyul-apply: output dir")
    parser.add_argument("--num-points", type=int, default=10,
                        help="nyul-train: interior landmark count")
    parser.add_argument("--p-low", type=float, default=1.0)
    parser.add_argument("--p-high", type=float, default=99.0)
    parser.add_argument("--features", type=Path, help="combat: feature CSV")
    parser.add_argument("--batch-col", help="combat: batch column name")
    parser.add_argument("--covariate-cols", default="",
                        help="combat: covariates to preserve, comma separated")
    parser.add_argument("--categorical-cols", default="",
                        help="combat: covariates declared categorical, comma separated")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point.

    Returns:
        Exit code: 0 success; 1 self-check failure; 2 argument or data error.
    """
    import sys

    args = build_parser().parse_args(argv)

    if args.command == "verify":
        checks = self_check()
        n_fail = len([1 for _, ok, _ in checks if not ok])
        for name, ok, msg in checks:
            print(f"{'[PASS]' if ok else '[FAIL]'} {name}: {msg}")
        print(f"summary: {'all checks passed' if n_fail == 0 else f'{n_fail} check(s) FAILED'}")
        return 0 if n_fail == 0 else 1

    # File-based commands share the chdir-safe IO of lumbarmri.io.
    from .io import read_image as _read_sitk
    from .io import write_image

    if args.command == "zscore":
        if not (args.image and args.output):
            print("error: --image and --output are required", file=sys.stderr)
            return 2
        img = _read_sitk(args.image)
        mask = None
        if args.mask is not None:
            mask = sitk.Cast(_read_sitk(args.mask) > 0, sitk.sitkUInt8)
        out = zscore(img, mask, clip_sigma=args.clip_sigma)
        write_image(out, args.output)
        print(f"wrote {args.output}")
        return 0

    if args.command == "hist-match":
        if not (args.image and args.reference and args.output):
            print("error: --image, --reference and --output are required", file=sys.stderr)
            return 2
        out = histogram_match(_read_sitk(args.image), _read_sitk(args.reference))
        write_image(out, args.output)
        print(f"wrote {args.output}")
        return 0

    if args.command == "nyul-train":
        from .preprocess import discover_cases, find_image_for, read_image_f32, read_mask

        if not (args.root and args.model_out):
            print("error: --root and --model-out are required", file=sys.stderr)
            return 2
        samples = []
        for _cid, _cdir, masks in discover_cases(args.root, args.case_list):
            for mp in masks:
                img = read_image_f32(find_image_for(mp))
                m = sitk.GetArrayViewFromImage(read_mask(mp)) > 0
                samples.append((sitk.GetArrayFromImage(img), m))
        if not samples:
            print("error: training set is empty", file=sys.stderr)
            return 2
        model = fit_nyul(samples, num_points=args.num_points,
                         p_low=args.p_low, p_high=args.p_high)
        model.save(args.model_out)
        print(f"trained on {len(samples)} image/mask pairs -> {args.model_out}")
        print(f"landmarks: {np.round(model.landmarks, 2)}")
        return 0

    if args.command == "nyul-apply":
        from .preprocess import discover_cases, find_image_for, read_image_f32, read_mask

        if not (args.root and args.model and args.output_dir):
            print("error: --root, --model and --output-dir are required", file=sys.stderr)
            return 2
        model = NyulModel.load(args.model)
        out_dir = args.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for _cid, _cdir, masks in discover_cases(args.root, args.case_list):
            for mp in masks:
                img = read_image_f32(find_image_for(mp))
                norm = apply_nyul(img, model, read_mask(mp))
                dst = out_dir / f"{mp.name.replace('.nii.gz', '')}_nyul.nii.gz"
                write_image(norm, dst)
                n += 1
        print(f"wrote {n} Nyul-normalized images -> {out_dir}")
        return 0

    if args.command == "combat":
        import pandas as pd

        if not (args.features and args.batch_col and args.output):
            print("error: --features, --batch-col and --output are required",
                  file=sys.stderr)
            return 2
        df = pd.read_csv(args.features)
        covars = [c.strip() for c in (args.covariate_cols or "").split(",") if c.strip()]
        cats = [c.strip() for c in (args.categorical_cols or "").split(",") if c.strip()]
        out = combat_harmonize(df, args.batch_col, covars, cats)
        out.to_csv(args.output, index=False, encoding="utf-8-sig")
        print(f"wrote {args.output} ({out.shape[0]} rows x {out.shape[1]} cols)")
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
