"""Golden-value tests for z-continuity splitting / relabeling."""

import numpy as np
import pytest
import SimpleITK as sitk

from lumbarmri.io import read_image, write_image
from lumbarmri.zsplit import (
    find_z_groups,
    relabel_z_continuity,
    split_output_name,
    strip_label_stem,
)


def test_find_z_groups_two_runs():
    arr = np.zeros((6, 3, 3), dtype=np.uint8)
    arr[1:3, :, :] = 1  # slices 1-2
    arr[4:6, :, :] = 1  # slices 4-5
    groups = find_z_groups(arr, target_label=1)
    assert [(g.start_z, g.end_z) for g in groups] == [(1, 2), (4, 5)]
    assert [g.n_slices for g in groups] == [2, 2]


def test_find_z_groups_empty_and_single_slice():
    assert find_z_groups(np.zeros((4, 2, 2), dtype=np.uint8), 1) == []
    arr = np.zeros((4, 2, 2), dtype=np.uint8)
    arr[2, 0, 0] = 1
    groups = find_z_groups(arr, 1)
    assert [(g.start_z, g.end_z) for g in groups] == [(2, 2)]


def test_relabel_z_continuity(tmp_path):
    arr = np.zeros((6, 3, 3), dtype=np.uint8)
    arr[0:2, :, :] = 5  # run 1 -> label 1
    arr[4:6, :, :] = 5  # run 2 -> label 2
    mask_path = tmp_path / "case_01_ME_label.nii.gz"
    write_image(sitk.GetImageFromArray(arr), mask_path)

    res = relabel_z_continuity(
        mask_path, output_dir=tmp_path, target_label=5, overwrite=True
    )
    out = sitk.GetArrayFromImage(read_image(res.output_path))
    assert set(np.unique(out).tolist()) == {0, 1, 2}
    assert out[0, 0, 0] == 1 and out[1, 0, 0] == 1
    assert out[4, 0, 0] == 2 and out[5, 0, 0] == 2
    assert [(g.start_z, g.end_z) for g in res.groups] == [(0, 1), (4, 5)]


def test_strip_label_stem():
    assert strip_label_stem("case_01_ME_label.nii.gz") == ("case_01_ME", "")
    assert strip_label_stem("case_01_ME_label2.nii.gz") == ("case_01_ME", "2")
    with pytest.raises(ValueError):
        strip_label_stem("case_01_ME_roi.nii.gz")


def test_split_output_name():
    assert split_output_name("case_01_ME_label.nii.gz", 1, "_label.nii.gz") == "case_01_ME_1_label.nii.gz"
    assert (
        split_output_name("case_01_ME_label2.nii.gz", 3, "_label.nii.gz")
        == "case_01_ME_label2_3_label.nii.gz"
    )
