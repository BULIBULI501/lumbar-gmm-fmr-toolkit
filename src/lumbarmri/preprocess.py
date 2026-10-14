"""N4ITK bias-field correction for lumbar paravertebral muscle MRI.

MRI brightness is not quantitative: coil sensitivity and field inhomogeneity
multiply the true signal by a smooth low-frequency field::

    I_observed(x) = I_true(x) * B(x)

N4ITK (Tustison 2010) fits the field and tissue intensities alternately with a
multi-resolution B-spline model, no tissue classification required.  This
module follows the official SimpleITK recipe ``N4BiasFieldCorrection.py``:
estimate the field on a *shrunk* image, rebuild it at full resolution with
``GetLogBiasFieldAsImage``, then divide ``original / exp(log_bias)``.  The
official example explicitly warns against iterating at full resolution (it
amplifies noise).

Anisotropic data (recommended path)
-----------------------------------
When slice thickness far exceeds in-plane pixel size (this study: 512x512x12,
10.9-15.5 mm slices vs 0.39 mm in-plane), a 3D N4 interpolates inter-slice
continuity that does not exist.  The bias field is fundamentally an *in-plane*
smooth field, so prefer :func:`correct_per_slice` (per-slice 2D N4) for such
data — the same logic behind IBSI's per-slice recommendation for thick
slices.  Both modes are kept; ``mode='both'`` supports sensitivity analysis.

Estimation mask (``mask_method``)
---------------------------------
``roi`` (default, the research convention)
    estimate the field inside the given ROI only, so that abdominal fat /
    body wall do not drag the B-spline control points away from the target
    muscle (measured on 3 cases x 2 regions: ROI masks cut CV ~5.6x more
    than whole-image Otsu for ME).
``otsu``
    whole-image Otsu threshold (official example default).  One correction
    serves every mask of the same image, at the price of weaker in-ROI
    correction.
``mean`` / ``percentile``
    intensity-threshold fallbacks for images without a usable ROI.

Pipeline order (do not reorder)::

    resample -> N4 bias-field correction -> intensity normalization ->
    discretization / feature extraction

N4 must precede normalization: it changes *relative* brightness inside one
image, while normalization statistics depend on the corrected grayscale.

Conventions frozen for this toolkit
-----------------------------------
* Correction formula: ``corrected = original / exp(log_bias)``; the returned
  field is in the **log** domain, never subtract it directly.
* ``roi_stats``-style grayscale statistics use ``ddof=0`` (population std),
  matching the study's ICC/FMR baselines.
* The 2D slice extraction always forwards the in-plane direction block
  ``D[0:2, 0:2]`` of the parent 3D image; a previous implementation dropped
  it (``GetImageFromArray`` images have identity direction, so a
  ``np.allclose`` guard was dead code and the physical rotation was lost).

The SimpleITK non-ASCII-path workaround (chdir + plain filename) lives in
:mod:`lumbarmri.io` and is used by the file-based entry points.

References
----------
- Sled JG, et al. IEEE TMI, 1998, 17(1):99-105. (N3)
- Tustison NJ, et al. IEEE TMI, 2010, 29(6):1310-1320. (N4ITK)
"""

from __future__ import annotations

import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk

from .io import read_image as _read_sitk
from .io import write_image
from .normalization import roi_stats

LOGGER = logging.getLogger(__name__)

Mode = Literal["2d", "3d", "both"]

#: Mask filename discovery pattern, consistent with gmm_segmentation.py
MASK_NAME_PATTERN = re.compile(r"^(?P<stem>.+)_label(?P<draw>\d*)\.nii\.gz$")
DEFAULT_IMG_SUFFIX = "_image.nii.gz"

TOL_EXACT = 1e-9
TOL_FLOAT = 1e-3

__all__ = [
    "N4Params",
    "N4Result",
    "N4Stats",
    "build_mask",
    "uniformity_metrics",
    "segmentation_change",
    "correct",
    "correct_3d",
    "correct_per_slice",
    "n4_correct",
    "read_image_f32",
    "read_mask",
    "synthetic_phantom",
    "discover_cases",
    "find_image_for",
    "filter_masks",
    "self_check",
]


# ---------------------------------------------------------------------------
# Configuration data structures
# ---------------------------------------------------------------------------


@dataclass
class N4Params:
    """N4ITK parameters (defaults = SimpleITK 2.5.3 filter defaults).

    Field names follow the ``simpleitk_n4tool`` base (``control_points``,
    ``fwhm``, ``histogram_bins``); the research script
    ``mri_preprocess_n4.py`` used ``n_control_points`` / ``bias_field_fwhm`` /
    ``n_histogram_bins`` for the same values.

    Attributes:
        iterations: max iterations per pyramid level; **vector length = number
            of levels**.
        control_points: B-spline control-point grid size; the actual grid is
            ``control_points - spline_order`` and must stay positive.
        spline_order: B-spline order.
        fwhm: Gaussian deconvolution FWHM of the bias field (spatial scale).
        histogram_bins: bins of the log-intensity histogram.
        convergence_threshold: per-level convergence criterion — coefficient
            of variation of the field difference map.
        wiener_filter_noise: Wiener filter noise estimate; Sled 1998's ``Z``
            is its square root (default 0.01 -> Z = 0.1).
        shrink: downsampling factor before field estimation. ``1`` = slowest
            and most accurate, ``4`` = faster with detail loss.
        mask_label: only voxels equal to this label enter the estimation.
    """

    iterations: Tuple[int, ...] = (50, 50, 50, 50)
    control_points: int = 4
    spline_order: int = 3
    fwhm: float = 0.15
    histogram_bins: int = 200
    convergence_threshold: float = 0.001
    wiener_filter_noise: float = 0.01
    shrink: int = 2
    mask_label: int = 1

    def validate(self) -> "N4Params":
        """Validate parameters; raise ``ValueError`` on any illegal value."""

        if not self.iterations or any(i < 1 for i in self.iterations):
            raise ValueError(f"iterations must be positive ints, got {self.iterations}")
        if self.control_points <= self.spline_order:
            raise ValueError(
                f"control_points({self.control_points}) must be > "
                f"spline_order({self.spline_order}), else the B-spline grid is empty"
            )
        if not 0 < self.fwhm < 1:
            raise ValueError(f"fwhm must lie in (0, 1), got {self.fwhm}")
        if self.shrink < 1:
            raise ValueError(f"shrink must be >= 1, got {self.shrink}")
        if self.convergence_threshold <= 0:
            raise ValueError("convergence_threshold must be positive")
        return self


@dataclass
class N4Result:
    """Correction result and QC metrics of one run."""

    corrected: sitk.Image
    log_bias_field: np.ndarray
    cv_before: float = float("nan")
    cv_after: float = float("nan")
    lr_ratio_before: float = float("nan")
    lr_ratio_after: float = float("nan")
    log_bias_min: float = float("nan")
    log_bias_max: float = float("nan")

    @property
    def cv_reduction_pct(self) -> float:
        """Relative CV change (negative = more uniform)."""

        if not (self.cv_before and np.isfinite(self.cv_before) and self.cv_before > 0):
            return float("nan")
        return 100.0 * (self.cv_after - self.cv_before) / self.cv_before

    def summary(self) -> Dict[str, float]:
        """Flatten to a dict for CSV export."""

        return {
            "cv_before": self.cv_before,
            "cv_after": self.cv_after,
            "cv_reduction_pct": self.cv_reduction_pct,
            "lr_ratio_before": self.lr_ratio_before,
            "lr_ratio_after": self.lr_ratio_after,
            "log_bias_min": self.log_bias_min,
            "log_bias_max": self.log_bias_max,
        }


@dataclass
class N4Stats:
    """Per-case N4 QC row (file-based entry points, batch CSV export).

    ``cv_reduction_pct`` in :meth:`as_row` is defined as
    ``100 * (before - after) / before`` — **positive = improvement**, matching
    the research script's batch summary convention.
    """

    path: str
    mode: str
    #: ROI grayscale coefficient of variation (before / after)
    cv_before: float = float("nan")
    cv_after: float = float("nan")
    #: ROI grayscale mean (before / after)
    mean_before: float = float("nan")
    mean_after: float = float("nan")
    #: log bias field range inside the ROI (> 0 spread = spatial inhomogeneity)
    log_bias_min: float = float("nan")
    log_bias_max: float = float("nan")
    #: left/right half ROI mean ratio (1.0 = no left-right gradient)
    lr_ratio_before: float = float("nan")
    lr_ratio_after: float = float("nan")
    #: ``mode='both'``: per-slice 2D reference metrics for sensitivity analysis
    cv_after_2d: float = float("nan")
    lr_ratio_after_2d: float = float("nan")
    #: wall-clock seconds of the correction
    seconds: float = float("nan")
    ok: bool = True
    message: str = ""

    def as_row(self) -> Dict[str, object]:
        return {
            "path": self.path,
            "mode": self.mode,
            "cv_before": self.cv_before,
            "cv_after": self.cv_after,
            "cv_reduction_pct": (
                100.0 * (self.cv_before - self.cv_after) / self.cv_before
                if self.cv_before and np.isfinite(self.cv_before) and self.cv_before > 0
                else float("nan")
            ),
            "mean_before": self.mean_before,
            "mean_after": self.mean_after,
            "log_bias_min": self.log_bias_min,
            "log_bias_max": self.log_bias_max,
            "lr_ratio_before": self.lr_ratio_before,
            "lr_ratio_after": self.lr_ratio_after,
            "cv_after_2d": self.cv_after_2d,
            "lr_ratio_after_2d": self.lr_ratio_after_2d,
            "seconds": self.seconds,
            "ok": self.ok,
            "message": self.message,
        }


# ---------------------------------------------------------------------------
# File-based helpers (chdir channel lives in lumbarmri.io)
# ---------------------------------------------------------------------------


def read_image_f32(path) -> sitk.Image:
    """Read an image as float32, recording the original pixel type.

    The original pixel type string is attached as attribute
    ``original_pixel_type`` *after* the cast: ``sitk.Cast`` returns a new
    object and a ``setattr`` done before the cast is lost (measured: the cast
    result does not carry the attribute).

    Args:
        path: image file path.

    Returns:
        float32 SimpleITK image with ``original_pixel_type`` attribute.
    """
    img = _read_sitk(Path(path))
    original_type = img.GetPixelIDTypeAsString()
    out = sitk.Cast(img, sitk.sitkFloat32)
    setattr(out, "original_pixel_type", original_type)
    return out


def read_mask(path) -> sitk.Image:
    """Read a mask and normalize to {0, 1} uint8.

    Different drawing batches store ROI voxels as 1 or 2 (``label`` in {0,1},
    ``label2`` in {0,1} or {0,2}); every voxel with ``mask > 0`` is ROI and is
    normalized to 1.
    """
    m = _read_sitk(Path(path))
    return sitk.Cast(m > 0, sitk.sitkUInt8)


def _array(img) -> np.ndarray:
    """sitk.Image -> ndarray; passthrough for ndarray inputs."""

    return img if isinstance(img, np.ndarray) else sitk.GetArrayFromImage(img)


def _mask_array(mask: Optional[sitk.Image], ref: sitk.Image) -> np.ndarray:
    if mask is None:
        return np.ones(_array(ref).shape, dtype=bool)
    m = _array(mask).astype(bool)
    if m.shape != _array(ref).shape:
        raise ValueError(
            f"mask shape {m.shape} does not match image shape {_array(ref).shape}"
        )
    return m


def build_mask(
    img: sitk.Image,
    method: str = "otsu",
    roi: Optional[sitk.Image] = None,
    percentile: float = 70.0,
) -> sitk.Image:
    """Build the N4 estimation mask, excluding air / low-signal background.

    Args:
        img: input image.
        method: ``otsu`` — Otsu threshold (official example default, good for
            images with clear foreground/background contrast);
            ``mean`` — above the whole-image mean (steadier for body images
            with dark ROIs);
            ``roi`` — use the given mask directly (recommended for
            paravertebral muscle work: estimate the field inside the ROI only,
            avoiding fat / abdominal contamination of the control points);
            ``percentile`` — above the given percentile.
        roi: ROI mask; required for ``method='roi'``.
        percentile: percentile (1-100) for ``method='percentile'``; for
            ``method='mean'`` with an ROI the threshold is the ROI mean.

    Returns:
        uint8 estimation mask.
    """
    if method == "roi":
        if roi is None:
            raise ValueError("method='roi' requires a roi mask")
        return sitk.Cast(roi > 0, sitk.sitkUInt8)
    if method == "otsu":
        return sitk.OtsuThreshold(img, 0, 1, 200)
    if method == "mean":
        # Whole-image mean via a SimpleITK operator; the ROI mean uses the
        # read-only-view statistics helper (3D label statistics filters are
        # unavailable in SimpleITK — see normalization.roi_stats).
        if roi is None:
            st = sitk.StatisticsImageFilter()
            st.Execute(img)
            thr = float(st.GetMean())
        else:
            thr = roi_stats(img, roi)[0]
        return sitk.Cast(img > thr, sitk.sitkUInt8)
    if method == "percentile":
        arr = _array(img)
        ref = arr if roi is None else arr[_array(roi) > 0]
        thr = float(np.percentile(ref, percentile))
        return sitk.Cast(img > thr, sitk.sitkUInt8)
    raise ValueError(f"unknown mask method: {method}")


# ---------------------------------------------------------------------------
# N4 bias-field correction core
# ---------------------------------------------------------------------------


def _configure(params: N4Params) -> sitk.N4BiasFieldCorrectionImageFilter:
    f = sitk.N4BiasFieldCorrectionImageFilter()
    f.SetMaximumNumberOfIterations(list(params.iterations))
    f.SetNumberOfControlPoints(params.control_points)
    f.SetSplineOrder(params.spline_order)
    f.SetBiasFieldFullWidthAtHalfMaximum(params.fwhm)
    f.SetNumberOfHistogramBins(params.histogram_bins)
    f.SetConvergenceThreshold(params.convergence_threshold)
    f.SetWienerFilterNoise(params.wiener_filter_noise)
    f.SetMaskLabel(params.mask_label)
    f.SetUseMaskLabel(True)
    return f


def _shrink_pair(
    image: sitk.Image, mask: sitk.Image, factor: int
) -> Tuple[sitk.Image, sitk.Image]:
    """Shrink image and mask together so voxels stay one-to-one.

    ``sitk.Shrink()`` only accepts ``(image, factors)`` and
    ``ShrinkImageFilter`` exposes neither ``SetOrder`` nor ``SetInterpolator``,
    so the interpolation cannot be pinned.  The shrunk mask is re-binarized
    with ``> 0``: if linear averaging happened underneath, boundary voxels are
    faded to fractions and ``> 0`` pulls them back — the shrunk mask then only
    *grows* relative to the original, which is directionally conservative
    (valid regions are never excluded from the estimation).  The image side
    stays float32 because N4 works in log space and uint8 would lose precision
    first.
    """
    if factor <= 1:
        return image, mask
    dims = [factor] * image.GetDimension()
    return (
        sitk.Shrink(sitk.Cast(image, sitk.sitkFloat32), dims),
        sitk.Cast(sitk.Shrink(mask, dims) > 0, sitk.sitkUInt8),
    )


def _slice_to_2d(
    volume: np.ndarray,
    index: int,
    spacing: Sequence[float],
    origin: Sequence[float],
    direction: Sequence[float],
) -> sitk.Image:
    """Extract slice ``index`` as a 2D image that keeps the physical metadata.

    Index convention (SimpleITK): ``GetSize()`` / ``GetSpacing()`` /
    ``GetOrigin()`` are ``(x, y, z)`` while ``GetArrayFromImage()`` returns
    ``(z, y, x)``; slice ``volume[index]`` therefore pairs with spacing/origin
    indices 0 and 1.

    The in-plane direction block **must** be forwarded explicitly: images
    created by ``sitk.GetImageFromArray`` always have identity direction, so
    dropping it loses the parent image's in-plane rotation/flip and the
    B-spline grid ends up on a distorted physical scale.  (A previous
    implementation guarded the forwarding with ``np.allclose(d, np.eye(2))``
    on the *new* image's direction — dead code; every real case in the source
    study carried a non-identity in-plane block.)
    """
    img2d = sitk.GetImageFromArray(np.ascontiguousarray(volume[index]))
    img2d.SetSpacing([float(spacing[0]), float(spacing[1])])
    img2d.SetOrigin([float(origin[0]), float(origin[1])])
    d3 = np.asarray(direction, dtype=np.float64).reshape(3, 3)
    img2d.SetDirection([float(x) for x in d3[0:2, 0:2].ravel()])
    return img2d


def uniformity_metrics(
    image: np.ndarray, mask: Optional[np.ndarray]
) -> Dict[str, float]:
    """Grayscale uniformity metrics: CV, left/right ratio, robust range.

    Definitions:
        cv: ROI grayscale coefficient of variation ``std / mean``; smaller is
            more uniform (the cross-case comparable quantity).
        lr_ratio: left-half ROI mean / right-half ROI mean; 1.0 = no
            left-right gradient (tracks low-frequency field residue).  For 3D
            input it is computed **per slice** and summarized by the median —
            per-slice N4 estimates independent fields, so pooling all slices
            would mix inter-slice differences into the metric.
        p90_p10: 90th minus 10th percentile grayscale inside the ROI, a robust
            dynamic-range measure.
    """
    vals = image[mask] if mask is not None else image.reshape(-1)
    if vals.size < 10:
        return {"cv": float("nan"), "lr_ratio": float("nan"), "p90_p10": float("nan")}

    base = {
        "cv": float(vals.std() / vals.mean()) if vals.mean() else float("nan"),
        "p90_p10": float(np.percentile(vals, 90) - np.percentile(vals, 10)),
    }
    if mask is None:
        return {**base, "lr_ratio": float("nan")}

    # Boolean indexing needs a full-shape boolean array (no broadcasting of a
    # (n, m) mask over (k, n, m) slices).
    h, w = mask.shape[-2:]
    left, right = np.zeros_like(mask), np.zeros_like(mask)
    left[..., : w // 2] = True
    right[..., w // 2 :] = True

    ratios: list = []
    if image.ndim == 3:
        for k in range(image.shape[0]):
            lm, rm = mask[k] & left[k], mask[k] & right[k]
            if lm.any() and rm.any():
                ratios.append(float(image[k][lm].mean() / image[k][rm].mean()))
    else:
        lm, rm = mask & left, mask & right
        if lm.any() and rm.any():
            ratios.append(float(image[lm].mean() / image[rm].mean()))

    return {**base, "lr_ratio": float(np.median(ratios)) if ratios else float("nan")}


def _finalize(
    corrected: sitk.Image,
    log_bias: np.ndarray,
    before: np.ndarray,
    mask: Optional[np.ndarray],
) -> N4Result:
    after = sitk.GetArrayFromImage(corrected)
    u0 = uniformity_metrics(before, mask)
    u1 = uniformity_metrics(after, mask)
    res = N4Result(
        corrected=corrected,
        log_bias_field=log_bias,
        cv_before=u0["cv"],
        cv_after=u1["cv"],
        lr_ratio_before=u0["lr_ratio"],
        lr_ratio_after=u1["lr_ratio"],
    )
    if mask is not None and log_bias.shape == mask.shape:
        res.log_bias_min = float(log_bias[mask].min())
        res.log_bias_max = float(log_bias[mask].max())
    return res


def correct_3d(
    image: sitk.Image,
    mask: sitk.Image,
    params: Optional[N4Params] = None,
) -> N4Result:
    """Whole-volume 3D N4 correction, for isotropic data.

    The field is estimated on the shrunk image and rebuilt at full resolution
    (official recipe) to avoid noise amplification from full-resolution
    iteration.
    """
    p = (params or N4Params()).validate()
    small_img, small_mask = _shrink_pair(image, mask, p.shrink)
    corrector = _configure(p)
    corrector.Execute(small_img, small_mask)
    # The field estimated on the shrunk image must be rebuilt at the original
    # resolution before dividing.
    log_bias = corrector.GetLogBiasFieldAsImage(image)
    corrected = sitk.Cast(image / sitk.Exp(log_bias), sitk.sitkFloat32)
    return _finalize(
        corrected,
        sitk.GetArrayFromImage(log_bias),
        sitk.GetArrayFromImage(image),
        sitk.GetArrayViewFromImage(mask) > 0,
    )


def correct_per_slice(
    image: sitk.Image,
    mask: sitk.Image,
    params: Optional[N4Params] = None,
    min_roi_voxels: int = 10,
) -> N4Result:
    """Per-slice 2D N4 correction (recommended for strongly anisotropic data).

    Args:
        image: 3D image.
        mask: estimation mask (same geometry).
        params: N4 parameters.
        min_roi_voxels: slices with fewer ROI voxels are skipped (kept at
            input intensity, zero log-bias) so empty slices cannot degenerate
            the histogram estimate.
    """
    p = (params or N4Params()).validate()
    arr = sitk.GetArrayFromImage(image)
    m = sitk.GetArrayViewFromImage(mask) > 0
    out = arr.astype(np.float32, copy=True)
    log_bias_stack = np.zeros(arr.shape, dtype=np.float32)
    spacing, origin = image.GetSpacing(), image.GetOrigin()
    direction = image.GetDirection()

    for k in range(arr.shape[0]):
        if m[k].sum() < min_roi_voxels:
            LOGGER.debug("slice %d has < %d ROI voxels, skipped", k, min_roi_voxels)
            continue
        img2d = _slice_to_2d(arr, k, spacing, origin, direction)
        mask2d = sitk.GetImageFromArray(np.ascontiguousarray(m[k].astype(np.uint8)))
        mask2d.CopyInformation(img2d)
        small_img, small_mask = _shrink_pair(img2d, mask2d, p.shrink)
        corrector = _configure(p)
        corrector.Execute(small_img, small_mask)
        lb = corrector.GetLogBiasFieldAsImage(img2d)
        out[k] = sitk.GetArrayFromImage(img2d / sitk.Exp(lb))
        log_bias_stack[k] = sitk.GetArrayFromImage(lb)

    corrected = sitk.GetImageFromArray(out)
    corrected.CopyInformation(image)
    return _finalize(corrected, log_bias_stack, arr, m)


def correct(
    image: sitk.Image,
    roi: Optional[sitk.Image] = None,
    mode: Mode = "2d",
    params: Optional[N4Params] = None,
    mask_method: str = "roi",
) -> N4Result:
    """N4 bias-field correction main entry (image level).

    Args:
        image: input image.
        roi: ROI mask.  With ``mask_method='roi'`` (default) it is *also* the
            estimation mask; with other mask methods it only scopes the QC
            statistics (CV, left/right ratio).
        mode: ``2d`` per-slice (recommended for anisotropic data) / ``3d``
            whole volume.  ``both`` is only accepted by :func:`n4_correct`.
        params: N4 parameters.
        mask_method: see :func:`build_mask`.  Default ``roi`` follows the
            research convention (field estimated inside the target muscle).
    """
    p = (params or N4Params()).validate()
    if mode not in ("2d", "3d"):
        raise ValueError(f"mode must be '2d' or '3d' here, got {mode!r}")
    est_mask = build_mask(image, mask_method, roi=roi)
    if mode == "2d":
        return correct_per_slice(image, est_mask, p)
    return correct_3d(image, est_mask, p)


def segmentation_change(
    arr_before: np.ndarray,
    arr_after: np.ndarray,
    mask: np.ndarray,
    n_components: int = 2,
    random_state: int = 42,
) -> Dict[str, float]:
    """Quantify how bias correction perturbs the GMM fat/muscle split.

    Uses the same frozen two-step convention as ``gmm_segmentation``:
    ``mask > 0`` grayscale -> StandardScaler -> ``GaussianMixture(K=2,
    random_state=42)`` -> fat = cluster with larger ``mean_gray``.

    Returns:
        dict with ``dice`` (fat-mask agreement before/after; 1.0 = the
        segmentation is unchanged), ``fm_ratio_before`` / ``fm_ratio_after``
        (the Fat-to-Muscle Ratio primary endpoint) and
        ``fat_volume_change_pct`` (relative fat-voxel change).
    """
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler

    vals_b = arr_before[mask].reshape(-1, 1).astype(np.float64)
    vals_a = arr_after[mask].reshape(-1, 1).astype(np.float64)
    if vals_b.shape[0] < n_components * 2:
        return {
            "dice": float("nan"),
            "fm_ratio_before": float("nan"),
            "fm_ratio_after": float("nan"),
            "fat_volume_change_pct": float("nan"),
        }

    def _split(vals: np.ndarray) -> Tuple[np.ndarray, float]:
        x = StandardScaler().fit_transform(vals)
        gmm = GaussianMixture(n_components=n_components, random_state=random_state)
        lab = gmm.fit_predict(x) + 1
        raw = vals.ravel()
        means = {c: raw[lab == c].mean() for c in np.unique(lab)}
        fat = max(means, key=means.get)
        return lab == fat, float(means[fat] / sum(means.values()))

    fat_b, fm_b = _split(vals_b)
    fat_a, fm_a = _split(vals_a)
    denom = fat_b.sum() + fat_a.sum()
    dice = float(2.0 * (fat_b & fat_a).sum() / denom) if denom else float("nan")
    vol_change = (
        100.0 * (fat_a.sum() - fat_b.sum()) / fat_b.sum() if fat_b.sum() else float("nan")
    )
    return {
        "dice": dice,
        "fm_ratio_before": fm_b,
        "fm_ratio_after": fm_a,
        "fat_volume_change_pct": float(vol_change),
    }


# ---------------------------------------------------------------------------
# File-based main entry
# ---------------------------------------------------------------------------


def n4_correct(
    img_path,
    mask=None,
    params: Optional[N4Params] = None,
    mode: Mode = "2d",
    mask_method: str = "roi",
) -> Tuple[sitk.Image, np.ndarray, N4Stats]:
    """N4 correction main entry (file level).

    Args:
        img_path: source image path (never modified).
        mask: ROI mask path.  Given ``mask_method='roi'`` (default) it is both
            the estimation mask and the QC statistics scope — recommended for
            paravertebral muscle ROIs so that abdominal fat does not enter the
            B-spline fit.
        params: N4 parameters.
        mode: ``3d`` whole volume; ``2d`` per-slice (recommended for the
            study's 10.9-15.5 mm slices); ``both`` additionally runs a 2D pass
            whose metrics go to ``stats.cv_after_2d`` / ``lr_ratio_after_2d``
            for sensitivity comparison — **the returned image is the 3D
            result**.
        mask_method: ``otsu`` / ``mean`` / ``percentile`` / ``roi``.

    Returns:
        ``(corrected image, log bias field array, N4Stats)``.

    Note:
        The field is in the **log** domain: the correction is
        ``corrected = original / exp(log_bias)``, do not subtract it.
    """
    p = (params or N4Params()).validate()
    img = read_image_f32(img_path)
    roi = read_mask(mask) if mask is not None else None
    if roi is not None:
        roi = sitk.Cast(roi, img.GetPixelID())
    n4_mask = build_mask(img, mask_method, roi=roi)

    arr0 = _array(img)
    m = _mask_array(roi, img)
    stats = N4Stats(path=str(img_path), mode=mode)
    u0 = uniformity_metrics(arr0, m)
    stats.cv_before = u0["cv"]
    stats.lr_ratio_before = u0["lr_ratio"]
    stats.mean_before = float(arr0[m].mean())

    t0 = time.perf_counter()
    if mode == "3d":
        result = correct_3d(img, n4_mask, p)
        corrected, lb = result.corrected, result.log_bias_field
    elif mode == "2d":
        result = correct_per_slice(img, n4_mask, p)
        corrected, lb = result.corrected, result.log_bias_field
    elif mode == "both":
        result = correct_3d(img, n4_mask, p)
        corrected, lb = result.corrected, result.log_bias_field
        alt = correct_per_slice(img, n4_mask, p)
        u_alt = uniformity_metrics(_array(alt.corrected), m)
        stats.cv_after_2d = u_alt["cv"]
        stats.lr_ratio_after_2d = u_alt["lr_ratio"]
    else:
        raise ValueError(f"unknown mode: {mode}")

    stats.seconds = time.perf_counter() - t0
    arr1 = _array(corrected)
    u1 = uniformity_metrics(arr1, m)
    stats.cv_after = u1["cv"]
    stats.lr_ratio_after = u1["lr_ratio"]
    stats.mean_after = float(arr1[m].mean())
    stats.log_bias_min = float(lb[m].min())
    stats.log_bias_max = float(lb[m].max())
    return corrected, lb, stats


# ---------------------------------------------------------------------------
# Case discovery (shared by the tree subcommands and normalization CLI)
# ---------------------------------------------------------------------------


def discover_cases(
    root, case_list: Optional[str] = None
) -> List[Tuple[str, Path, List[Path]]]:
    """Discover ``(case_id, case dir, [mask paths])``.

    Handles two layouts: flat (``{id}_image.nii.gz`` + ``{id}_{L,ME,P}_label``
    files in one directory) and nested (``<root>/{case_id}/...``).
    ``case_list`` is a text file with one case id per line.
    """
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"data root does not exist: {root}")
    wanted: Optional[set] = None
    if case_list:
        wanted = {
            ln.strip()
            for ln in Path(case_list).read_text(encoding="utf-8").splitlines()
            if ln.strip()
        }

    cases: List[Tuple[str, Path, List[Path]]] = []
    subdirs = [d for d in sorted(root.iterdir()) if d.is_dir()]
    for d in subdirs:
        if wanted is not None and d.name not in wanted:
            continue
        if not any(d.glob("*_image.nii.gz")):
            continue
        masks = sorted(
            m for m in d.glob("*_label*.nii.gz") if MASK_NAME_PATTERN.match(m.name)
        )
        if masks:
            cases.append((d.name, d, masks))
    if wanted is not None and not cases:
        # Flat layout: masks sit next to images.
        masks = [
            m
            for m in sorted(root.glob("*_label*.nii.gz"))
            if MASK_NAME_PATTERN.match(m.name)
        ]
        for m in masks:
            cid = m.name.split("_label")[0]
            if cid in wanted:
                img = root / f"{cid}{DEFAULT_IMG_SUFFIX}"
                if img.exists():
                    cases.append((cid, root, [m]))
    return cases


def find_image_for(mask_path: Path) -> Path:
    """Locate the image matching a mask's stem prefix (longest prefix wins)."""
    stem = MASK_NAME_PATTERN.match(mask_path.name).group("stem")
    cands = []
    for img in mask_path.parent.glob("*_image.nii.gz"):
        if img.name[: -len(DEFAULT_IMG_SUFFIX)].startswith(stem.split("_")[0]):
            cands.append(img)
    if not cands:
        raise FileNotFoundError(
            f"no image matching {mask_path.name} in {mask_path.parent}"
        )
    return max(cands, key=lambda p: len(p.name))


def filter_masks(
    masks: Sequence[Path],
    regions: Optional[Sequence[str]] = None,
    draw: Optional[Sequence[int]] = None,
) -> List[Path]:
    """Filter masks by region and drawing batch.

    Naming: ``{id}_{region}_label{,2}.nii.gz`` with region in ``L`` / ``ME`` /
    ``P`` and drawing batch ``label`` (first) or ``label2`` (second).

    Args:
        masks: candidate mask paths.
        regions: keep these regions; ``None`` = keep all.  CLI spelling:
            ``ME,P``.
        draw: keep these drawing batches; ``None`` = keep all.  ``1`` ->
            ``label``, ``2`` -> ``label2``.  The bare ``_label`` suffix
            captures the empty string in the regex, so it is normalized to
            ``"1"`` before matching.
    """
    region_set = {r.strip().upper() for r in regions} if regions else None
    draw_set = {str(d) for d in draw} if draw else None

    out: List[Path] = []
    for mp in masks:
        m = MASK_NAME_PATTERN.match(mp.name)
        if not m:
            continue
        stem = m.group("stem")
        draw_tag = m.group("draw") or "1"
        # Region = stem minus the leading case-id segment.
        parts = stem.split("_", 1)
        region = parts[1] if len(parts) > 1 else ""

        if region_set is not None and region.upper() not in region_set:
            continue
        if draw_set is not None and draw_tag not in draw_set:
            continue
        out.append(mp)
    return out


# ---------------------------------------------------------------------------
# Synthetic phantom + self-check
# ---------------------------------------------------------------------------


def synthetic_phantom(
    two_tissue: bool, seed: int = 0, shape: Tuple[int, int, int] = (6, 96, 96)
) -> Tuple[sitk.Image, sitk.Image, np.ndarray]:
    """Build a synthetic phantom with known grayscale and known bias field.

    Args:
        two_tissue: ``False`` = homogeneous ROI (single grayscale) — pure
            probe of *field-estimation accuracy*, where all in-ROI variance
            comes from noise and the field, so CV and the left/right ratio are
            clean field metrics.  ``True`` = add a fat (high) blob over a
            muscle (mid) background — probes the correction's effect on the
            **GMM fat/muscle split**.
        seed: noise seed.
        shape: ``(z, y, x)`` volume shape.

    Returns:
        ``(image, mask, true log bias)``; the true field is mean-removed to
        match N4's internal zero-mean normalization of the estimated field.
    """
    rng = np.random.default_rng(seed)
    ny, nx = shape[1], shape[2]
    yy, xx = np.mgrid[0:ny, 0:nx]
    roi = ((yy - ny / 2) ** 2 / ((ny * 0.35) ** 2)
           + (xx - nx / 2) ** 2 / ((nx * 0.40) ** 2)) <= 1

    sl2d = np.full((ny, nx), 400.0)
    if two_tissue:
        fat = roi & (((yy - ny * 0.42) ** 2 / ((ny * 0.13) ** 2)
                      + (xx - nx * 0.54) ** 2 / ((nx * 0.15) ** 2)) <= 1)
        sl2d[fat] = 780.0
    sl2d += rng.normal(0, 3, (ny, nx))

    # Known multiplicative field: monotone x gradient 1.0 -> 1.45 (receiver
    # coil sensitivity profile).  **Deliberately not symmetrized** — a
    # symmetric field makes the two half means equal, pinning the LR metric at
    # 1 so it cannot detect the field at all (an early self-check failure
    # mode).
    grad = 1.0 + 0.45 * (xx / (nx - 1.0))
    bias2d = grad[None, :, :]

    vol = np.where(roi[None, :, :], sl2d[None, :, :], 50.0)
    biased = np.clip(
        vol * np.repeat(bias2d, shape[0], axis=0) + rng.normal(0, 2, shape), 1, None
    ).astype(np.float32)

    # N4 normalizes the field to zero mean internally; compare on the same
    # basis.
    true_log = np.log(bias2d)
    true_log -= true_log.mean(axis=(1, 2), keepdims=True)

    img = sitk.GetImageFromArray(biased)
    img.SetSpacing([12.0, 0.4, 0.4])  # reproduces the project's z-anisotropy ratio
    # Inject a non-identity in-plane direction block, matching the real study
    # data (measured: every case carried in-plane rotation/flip; e.g. one case
    # had det(D[0:2,0:2]) = 0.8015).
    d3 = np.eye(3)
    d3[0, 0], d3[0, 1] = 0.9992, 0.0384
    d3[1, 0], d3[1, 1] = -0.0234, 0.8013
    img.SetDirection([float(x) for x in d3.ravel()])
    mask = sitk.Cast(
        sitk.GetImageFromArray(np.repeat(roi[None, :, :], shape[0], 0).astype(np.uint8)),
        sitk.sitkUInt8,
    )
    mask.CopyInformation(img)
    return img, mask, true_log


def self_check() -> List[Tuple[str, bool, str]]:
    """Synthetic self-check (anchors A/B/D2 of the research verify command).

    A. **Field-estimation accuracy** (homogeneous phantom): inject a known log
       bias, compare the N4 estimate against it via the residual std inside
       the ROI (threshold 0.02) and require in-ROI CV not to increase.
    B. **Correction preserves the segmentation** (two-tissue phantom): GMM
       fat/muscle Dice before/after >= 0.95 in both modes — the criterion for
       "no need to re-baseline the ICC study".
    D2. **Direction propagation**: 2D slices must inherit the parent's
       in-plane ``D[0:2, 0:2]`` block and must not degenerate to identity.

    Note:
        An earlier criterion ("in-ROI CV must drop") is *falsely negative* on
        two-tissue ROIs: the 400-vs-780 tissue contrast dominates the
        variance, so a correctly working N4 looks like "no improvement".  That
        is why A uses the homogeneous phantom and B uses the Dice criterion.

    Returns:
        [(check name, passed, detail)] list.
    """
    results: List[Tuple[str, bool, str]] = []

    def record(name: str, ok: bool, msg: str) -> None:
        results.append((name, bool(ok), msg))

    params = N4Params()

    # ---- A. field estimation accuracy (homogeneous phantom) --------------
    img, mask, true_log = synthetic_phantom(two_tissue=False, seed=0)
    m = _mask_array(mask, img)
    u0 = uniformity_metrics(_array(img), m)
    for mode, runner in (("3d", correct_3d), ("2d", correct_per_slice)):
        res = runner(img, mask, params)
        err = float((res.log_bias_field - true_log)[m].std())
        u1 = uniformity_metrics(_array(res.corrected), m)
        record(f"A/{mode}: log-bias residual std <= 0.02", err <= 0.02,
               f"residual_std={err:.5f}")
        record(f"A/{mode}: in-ROI CV not increased", u1["cv"] <= u0["cv"],
               f"CV {u0['cv']:.5f} -> {u1['cv']:.5f}")

    # ---- B. segmentation stability (two-tissue phantom) ------------------
    img2, mask2, _ = synthetic_phantom(two_tissue=True, seed=42)
    m2 = _mask_array(mask2, img2)
    for mode, runner in (("3d", correct_3d), ("2d", correct_per_slice)):
        res = runner(img2, mask2, params)
        seg = segmentation_change(_array(img2), _array(res.corrected), m2)
        record(f"B/{mode}: GMM fat-mask Dice >= 0.95", seg["dice"] >= 0.95,
               f"Dice={seg['dice']:.4f}, FMR {seg['fm_ratio_before']:.4f} -> "
               f"{seg['fm_ratio_after']:.4f}, fat vol "
               f"{seg['fat_volume_change_pct']:+.2f}%")

    # ---- D2. direction propagation --------------------------------------
    d3 = np.array(img2.GetDirection()).reshape(3, 3)
    expect = d3[0:2, 0:2].ravel()
    got = np.array(
        _slice_to_2d(
            _array(img2), 1, img2.GetSpacing(), img2.GetOrigin(), img2.GetDirection()
        ).GetDirection()
    )
    record("D2: 2D slice inherits the 3D in-plane Direction",
           bool(np.allclose(expect, got, atol=1e-6)),
           f"expected={np.round(expect, 4).tolist()} got={np.round(got, 4).tolist()}")
    record("D2: 2D Direction is not degenerate identity",
           not bool(np.allclose(got, np.eye(2).ravel())),
           f"got={np.round(got, 4).tolist()}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser():
    """Build the argparse parser for the preprocess module CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="N4ITK bias-field correction (2D per-slice / 3D whole-volume)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  verify
  single --image x_image.nii.gz --mask x_ME_label.nii.gz --output-dir out \\
         --mode 2d --stats-out x_stats.csv
  tree   --root ./cases --output-dir out/n4 --mode 2d --only-regions ME,P
  qc     --image x_image.nii.gz --mask x_ME_label.nii.gz
""",
    )
    parser.add_argument(
        "command", choices=["verify", "single", "tree", "tree-single", "qc"],
        help="verify=self-check; single=one pair; tree=batch per-mask; "
             "tree-single=batch one correction per case; qc=uniformity+GMM report",
    )
    parser.add_argument("--image", type=Path, help="single/qc: image file")
    parser.add_argument("--mask", type=Path, help="single/qc: ROI mask file")
    parser.add_argument("--root", type=Path, help="tree/tree-single: data root")
    parser.add_argument("--case-list", type=Path,
                        help="tree/tree-single: text file of case ids (one per line)")
    parser.add_argument("--output-dir", type=Path, help="single/tree/tree-single: output dir")
    parser.add_argument("--mode", default="2d", choices=["3d", "2d", "both"],
                        help="correction mode (default: %(default)s)")
    parser.add_argument("--mask-method", default="roi",
                        choices=["roi", "otsu", "mean", "percentile"],
                        help="estimation mask construction (default: %(default)s)")
    parser.add_argument("--iterations", default="50,50,50,50",
                        help="max iterations per pyramid level, comma separated")
    parser.add_argument("--shrink", type=int, default=2,
                        help="downsample factor for field estimation")
    parser.add_argument("--control-points", type=int, default=4,
                        help="B-spline control points")
    parser.add_argument("--zscore", action="store_true",
                        help="also write ROI z-score normalized images")
    parser.add_argument("--save-log-bias", action="store_true",
                        help="single: also write the log bias field")
    parser.add_argument("--stats-out", type=Path,
                        help="single/qc: stats CSV path")
    parser.add_argument("--only-regions", default="",
                        help="tree/tree-single: only these regions, comma separated (ME,P)")
    parser.add_argument("--only-draw", default="",
                        help="tree/tree-single: only these drawing batches: 1=label, 2=label2")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point.

    Returns:
        Exit code: 0 success; 1 check/QC failure; 2 argument or data error.
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

    def _params() -> N4Params:
        return N4Params(
            iterations=tuple(int(x) for x in args.iterations.split(",")),
            shrink=args.shrink,
            control_points=args.control_points,
        )

    if args.command == "single":
        if args.output_dir is None:
            print("error: --output-dir is required", file=sys.stderr)
            return 2
        corrected, lb, stats = n4_correct(
            args.image, mask=args.mask, params=_params(),
            mode=args.mode, mask_method=args.mask_method,
        )
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        # Name outputs after the **mask** when given, so one image with
        # several ROI masks (ME/P/L) never overwrites itself.
        base = (args.mask or args.image).name.replace(".nii.gz", "")
        out_path = write_image(corrected, out_dir / f"{base}_n4.nii.gz")
        LOGGER.info("wrote %s", out_path)
        if args.save_log_bias:
            lb_img = sitk.GetImageFromArray(np.asarray(lb, dtype=np.float32))
            lb_img.CopyInformation(corrected)
            write_image(sitk.Cast(lb_img, sitk.sitkFloat32),
                        out_dir / f"{base}_logbias.nii.gz")
        if args.zscore:
            from .normalization import zscore

            zs = zscore(corrected, read_mask(args.mask) if args.mask else None)
            write_image(zs, out_dir / f"{base}_n4_zs.nii.gz")
        row = stats.as_row()
        for k, v in row.items():
            print(f"{k}: {v}")
        if args.stats_out:
            _write_stats_csv([row], args.stats_out)
        return 0 if stats.ok else 1

    if args.command in ("tree", "tree-single"):
        return _run_tree(args, single=(args.command == "tree-single"))

    if args.command == "qc":
        img = read_image_f32(args.image)
        m = (
            _mask_array(read_mask(args.mask), img)
            if args.mask
            else np.ones(_array(img).shape, bool)
        )
        a0 = _array(img)
        rows = []
        for mode in ("3d", "2d"):
            corrected, _, stats = n4_correct(
                args.image, mask=args.mask, params=_params(),
                mode=mode, mask_method=args.mask_method,
            )
            a1 = _array(corrected)
            row = stats.as_row()
            row.update(
                {f"seg_{k}": v for k, v in segmentation_change(a0, a1, m).items()}
            )
            rows.append(row)
            print(f"--- mode={mode} ---")
            for k, v in row.items():
                print(f"{k}: {v}")
        if args.stats_out:
            _write_stats_csv(rows, args.stats_out)
        return 0

    return 2


def _write_stats_csv(rows: List[Dict[str, object]], path: Path) -> None:
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    LOGGER.info("stats table: %s", path)


def _run_tree(args, single: bool) -> int:
    """Batch driver for the ``tree`` / ``tree-single`` subcommands.

    ``tree``: one correction **per mask** (each ROI estimates its own field —
    the paper-grade path).  ``tree-single``: one correction **per case** with
    a whole-image Otsu field (``mask_method`` fixed to ``otsu``) shared by all
    ROI masks of that case — fewer outputs, weaker in-ROI correction
    (measured: ROI-mask CV reduction ~5.6x larger than Otsu for ME).
    """
    import pandas as pd

    if args.output_dir is None:
        print("error: --output-dir is required", file=sys.stderr)
        return 2
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = discover_cases(args.root, args.case_list)
    if not cases:
        LOGGER.error("no cases found: root=%s case_list=%s", args.root, args.case_list)
        return 2
    regions = [r for r in (args.only_regions or "").split(",") if r.strip()]
    draws = [int(d) for d in (args.only_draw or "").split(",") if d.strip()]
    rows: List[Dict[str, object]] = []

    if single:
        n_cases = 0
        for cid, _cdir, masks in cases:
            sel = filter_masks(masks, regions, draws)
            if not sel:
                continue
            n_cases += 1
            img_p = find_image_for(sel[0])
            case_out = out_dir / cid
            case_out.mkdir(parents=True, exist_ok=True)
            try:
                corrected, _, st = n4_correct(
                    img_p, mask=None, params=_params_from(args),
                    mode=args.mode, mask_method="otsu",
                )
                stem = img_p.name.replace(".nii.gz", "")
                write_image(corrected, case_out / f"{stem}_n4.nii.gz")
                rows.append({"case": cid, "stage": "n4_otsu", **st.as_row()})
                if args.zscore:
                    from .normalization import zscore

                    write_image(zscore(corrected, None),
                                case_out / f"{stem}_n4_zs.nii.gz")
            except Exception as exc:  # keep the batch going on single-case failure
                LOGGER.error("failed %s N4: %s", cid, exc)
                rows.append({"case": cid, "stage": "n4_otsu", "ok": False,
                             "message": f"{type(exc).__name__}: {exc}"})
                continue
            for mp in sel:
                try:
                    mk = read_mask(mp)
                    mk_name = mp.name.replace("_label.nii.gz", "_mask.nii.gz")
                    write_image(mk, case_out / mk_name)
                    if args.zscore:
                        from .normalization import zscore

                        base = mp.name.replace(".nii.gz", "")
                        write_image(zscore(corrected, mk),
                                    case_out / f"{base}_n4_zs.nii.gz")
                    m_bool = _array(mk).astype(bool)
                    u0 = uniformity_metrics(_array(read_image_f32(img_p)), m_bool)
                    u1 = uniformity_metrics(_array(corrected), m_bool)
                    seg = segmentation_change(
                        _array(read_image_f32(img_p)), _array(corrected), m_bool
                    )
                    rows.append({
                        "case": cid, "stage": "roi_zs",
                        "region": mp.name.split("_")[1],
                        "cv_before": u0["cv"], "cv_after": u1["cv"],
                        "lr_ratio_before": u0["lr_ratio"],
                        "lr_ratio_after": u1["lr_ratio"],
                        "seg_dice": seg["dice"],
                        "seg_fm_ratio_before": seg["fm_ratio_before"],
                        "seg_fm_ratio_after": seg["fm_ratio_after"],
                        "seg_fat_volume_change_pct": seg["fat_volume_change_pct"],
                        "ok": True, "message": "",
                    })
                except Exception as exc:
                    LOGGER.error("failed %s %s: %s", cid, mp.name, exc)
                    rows.append({"case": cid, "stage": "roi_zs", "ok": False,
                                 "message": f"{type(exc).__name__}: {exc}"})
        if not rows:
            LOGGER.error("nothing processed after filtering")
            return 2
        _write_stats_csv(rows, out_dir / "single_image_stats.csv")
        LOGGER.info("done %d cases -> %s", n_cases, out_dir)
        return 0 if all(r.get("ok", True) for r in rows) else 1

    # --- tree: one correction per mask -----------------------------------
    total = 0
    for cid, _cdir, masks in cases:
        for mp in filter_masks(masks, regions, draws):
            total += 1
            try:
                img_p = find_image_for(mp)
                corrected, _, st = n4_correct(
                    img_p, mask=mp, params=_params_from(args),
                    mode=args.mode, mask_method=args.mask_method,
                )
                dst = out_dir / f"{mp.name.replace('.nii.gz', '')}_n4.nii.gz"
                write_image(corrected, dst)
                if args.zscore:
                    from .normalization import zscore

                    zs = zscore(corrected, read_mask(mp))
                    write_image(zs, dst.with_name(
                        dst.name.replace("_n4.nii.gz", "_n4_zs.nii.gz")))
            except Exception as exc:
                st = N4Stats(path=str(mp), mode=args.mode, ok=False,
                             message=f"{type(exc).__name__}: {exc}")
                LOGGER.error("failed %s: %s", mp.name, st.message)
            rows.append({"case": cid, "mask": mp.name, **st.as_row()})
            LOGGER.info("%s cv %.4f -> %.4f", mp.name, st.cv_before, st.cv_after)
    if total == 0:
        LOGGER.error("no masks after filtering: regions=%r draw=%r",
                     args.only_regions, args.only_draw)
        return 2
    _write_stats_csv(rows, out_dir / "n4_stats.csv")
    df = pd.DataFrame(rows)
    ok = df[df.ok == True]  # noqa: E712
    if not ok.empty:
        LOGGER.info(
            "done %d/%d; in-ROI CV median %.4f -> %.4f; |LR-1| median %.4f -> %.4f",
            len(ok), len(df), ok.cv_before.median(), ok.cv_after.median(),
            (ok.lr_ratio_before - 1).abs().median(),
            (ok.lr_ratio_after - 1).abs().median(),
        )
    return 0 if df.ok.all() else 1


def _params_from(args) -> N4Params:
    return N4Params(
        iterations=tuple(int(x) for x in args.iterations.split(",")),
        shrink=args.shrink,
        control_points=args.control_points,
    )


if __name__ == "__main__":
    raise SystemExit(main())
