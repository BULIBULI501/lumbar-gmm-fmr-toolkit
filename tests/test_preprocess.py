"""Golden-value tests for N4 correction and mask construction."""

import numpy as np
import pytest
import SimpleITK as sitk

from lumbarmri.preprocess import N4Params, build_mask, correct, synthetic_phantom


def _arr(img):
    return sitk.GetArrayFromImage(img)


def test_per_slice_n4_reduces_cv():
    """On the HOMOGENEOUS phantom the per-slice N4 must flatten the known bias
    field: CV after < 0.02 (reference run: 0.0086) and below CV before.

    Note: the CV criterion is only valid on the homogeneous phantom -- on the
    two-tissue phantom the 400-vs-780 tissue contrast dominates the variance
    (frozen convention, see preprocess.self_check note).
    """
    img, mask, _true_log = synthetic_phantom(two_tissue=False, seed=0, shape=(6, 96, 96))

    res = correct(img, roi=mask, mode="2d", params=N4Params(), mask_method="roi")
    assert np.isfinite(res.cv_before) and np.isfinite(res.cv_after)
    assert res.cv_after < res.cv_before
    assert res.cv_after < 0.02, f"CV after per-slice N4 too high: {res.cv_after}"
    # cv_reduction_pct = 100*(after-before)/before: negative = more uniform
    assert res.cv_reduction_pct < -50.0, "bias field barely reduced"


def test_3d_n4_runs_and_finite():
    img, mask, _ = synthetic_phantom(two_tissue=False, seed=0, shape=(6, 96, 96))
    res = correct(img, roi=mask, mode="3d", params=N4Params(), mask_method="roi")
    assert np.isfinite(res.cv_after)
    assert res.cv_after < res.cv_before


def test_build_mask_methods_are_binary():
    img, roi, _ = synthetic_phantom(two_tissue=True, seed=0, shape=(4, 32, 32))
    for method in ("otsu", "mean", "percentile"):
        vals = set(np.unique(_arr(build_mask(img, method=method))).tolist())
        assert vals <= {0, 1}, f"{method} mask not binary: {vals}"

    m_roi = build_mask(img, method="roi", roi=roi)
    np.testing.assert_array_equal(_arr(m_roi), (_arr(roi) > 0).astype(np.uint8))


def test_shape_mismatch_mask_raises():
    img, mask, _ = synthetic_phantom(two_tissue=True, seed=0, shape=(4, 32, 32))
    bad = sitk.GetImageFromArray(_arr(mask)[:, :16, :])
    with pytest.raises(Exception):
        correct(img, roi=bad, mode="2d", params=N4Params(), mask_method="roi")
