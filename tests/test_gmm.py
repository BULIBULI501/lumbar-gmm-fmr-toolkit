"""Golden-value tests for the frozen GMM conventions."""

from pathlib import Path

import numpy as np
import pytest

from lumbarmri.gmm_segmentation import GMM_N_COMPONENTS, RoiResult, run_gmm_on_roi


def _roi_result(clusters, n, degenerate):
    return RoiResult(
        image_path=Path("x_image.nii.gz"),
        mask_path=Path("x_ME_label.nii.gz"),
        output_mask=Path("out_mask.nii.gz"),
        stats_csv=Path("out_stats.csv"),
        clusters=clusters,
        roi_voxels=n,
        degenerate=degenerate,
    )


def test_two_level_phantom_fm_ratio_golden():
    """30 voxels @100 + 70 voxels @200 -> FM_ratio exactly 0.7.

    The fat cluster is the largest-mean_gray cluster (frozen convention).
    """
    img = np.zeros((2, 5, 10), dtype=np.float32)  # 100 voxels
    flat = img.reshape(-1)
    flat[:30] = 100.0
    flat[30:] = 200.0
    mask = np.ones((2, 5, 10), dtype=np.uint8)

    clusters, n_vox, degenerate = run_gmm_on_roi(img, mask, spacing=(1.0, 1.0, 1.0))
    assert n_vox == 100
    assert not degenerate
    assert len(clusters) == GMM_N_COMPONENTS

    fat = max(clusters, key=lambda c: c.mean_gray)
    assert fat.count == 70, "fat cluster must be the larger-mean gray level"
    assert abs(fat.mean_gray - 200.0) < 1e-6

    res = _roi_result(clusters, n_vox, degenerate)
    assert abs(res.fm_ratio - 0.7) < 1e-9


def test_degenerate_single_gray_level():
    """A single gray level falls back to one cluster, FM_ratio = 1.0."""
    img = np.full((1, 4, 4), 128.0, dtype=np.float32)
    mask = np.ones((1, 4, 4), dtype=np.uint8)

    clusters, n_vox, degenerate = run_gmm_on_roi(img, mask, spacing=(1.0, 1.0, 1.0))
    assert degenerate
    assert n_vox == 16

    res = _roi_result(clusters, n_vox, degenerate)
    assert res.fm_ratio == 1.0


def test_illegal_mask_values_raise():
    img = np.zeros((1, 2, 2), dtype=np.float32)
    mask = np.array([[[0, 3], [1, 0]]], dtype=np.uint8)
    with pytest.raises(ValueError):
        run_gmm_on_roi(img, mask, spacing=(1.0, 1.0, 1.0))


def test_shape_mismatch_raises():
    img = np.zeros((1, 2, 2), dtype=np.float32)
    mask = np.ones((1, 2, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        run_gmm_on_roi(img, mask, spacing=(1.0, 1.0, 1.0))


def test_volume_scaling_via_spacing():
    """volume_mm3 = count x voxel volume; FM_ratio is spacing-invariant."""
    img = np.zeros((2, 5, 10), dtype=np.float32)  # 100 voxels
    flat = img.reshape(-1)
    flat[:30] = 100.0
    flat[30:] = 200.0
    mask = np.ones((2, 5, 10), dtype=np.uint8)

    clusters, n_vox, degenerate = run_gmm_on_roi(img, mask, spacing=(0.4, 0.4, 12.0))
    voxel_vol = 0.4 * 0.4 * 12.0
    for c in clusters:
        assert abs(c.volume_mm3 - c.count * voxel_vol) < 1e-9
    res = _roi_result(clusters, n_vox, degenerate)
    assert abs(res.fm_ratio - 0.7) < 1e-9
