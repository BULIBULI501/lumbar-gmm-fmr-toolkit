"""Golden-value tests for intensity normalization (z-score / Nyul)."""

import numpy as np
import SimpleITK as sitk

from lumbarmri.normalization import NyulModel, apply_nyul, fit_nyul, roi_stats, zscore

TOL = 1e-3


def test_zscore_invariants():
    rng = np.random.RandomState(0)
    arr = rng.normal(loc=500.0, scale=80.0, size=(3, 8, 8)).astype(np.float32)
    mask_arr = np.zeros_like(arr, dtype=np.uint8)
    mask_arr[:, 2:6, 2:6] = 1

    img = sitk.GetImageFromArray(arr)
    mask = sitk.GetImageFromArray(mask_arr)
    out = sitk.GetArrayFromImage(zscore(img, mask))

    inside = out[mask_arr > 0]
    outside = out[mask_arr == 0]
    assert abs(inside.mean()) < 1e-6
    assert abs(inside.std(ddof=0) - 1.0) < TOL
    assert np.all(outside == 0)


def test_zscore_clamp_at_clip_sigma():
    rng = np.random.RandomState(1)
    arr = rng.normal(loc=100.0, scale=10.0, size=(1, 6, 6)).astype(np.float32)
    arr[0, 0, 0] = 1000.0  # extreme outlier
    mask_arr = np.ones_like(arr, dtype=np.uint8)

    out = sitk.GetArrayFromImage(
        zscore(sitk.GetImageFromArray(arr), sitk.GetImageFromArray(mask_arr), clip_sigma=3.0)
    )
    assert abs(float(out.max()) - 3.0) < TOL, "outlier must clamp at exactly +3 sigma"
    assert float(out.min()) >= -3.0 - 1e-6


def test_zscore_degenerate_roi_keeps_raw_gray():
    arr = np.full((1, 4, 4), 77.0, dtype=np.float32)
    mask_arr = np.ones_like(arr, dtype=np.uint8)
    out = sitk.GetArrayFromImage(
        zscore(sitk.GetImageFromArray(arr), sitk.GetImageFromArray(mask_arr))
    )
    np.testing.assert_allclose(out, arr, rtol=0, atol=1e-6)


def test_roi_stats_ddof0():
    arr = np.array([[[1.0, 3.0]]], dtype=np.float32)  # mean 2, population sd 1
    mean, sd = roi_stats(arr, None)
    assert abs(mean - 2.0) < 1e-9
    assert abs(sd - 1.0) < 1e-9, "roi_stats must use ddof=0"


def test_nyul_monotonic_and_roundtrip(tmp_path):
    rng = np.random.RandomState(2)
    samples = []
    for scale in (1.0, 2.5):
        arr = (rng.normal(loc=50 * scale, scale=10 * scale, size=(2, 12, 12))).astype(np.float32)
        samples.append((arr, None))

    model = fit_nyul(samples, num_points=10)

    probe = np.linspace(0.0, 150.0, 64, dtype=np.float32).reshape(1, 8, 8)
    img = sitk.GetImageFromArray(probe)
    out = sitk.GetArrayFromImage(apply_nyul(img, model, None)).reshape(-1)

    # monotonic: larger input -> larger output
    order = np.argsort(probe.reshape(-1), kind="stable")
    assert np.all(np.diff(out[order]) > -1e-6)

    # npz round-trip: saved and reloaded models agree exactly
    path = tmp_path / "nyul.npz"
    model.save(path)
    out2 = sitk.GetArrayFromImage(apply_nyul(img, NyulModel.load(path), None)).reshape(-1)
    np.testing.assert_allclose(out, out2, rtol=0, atol=1e-9)
