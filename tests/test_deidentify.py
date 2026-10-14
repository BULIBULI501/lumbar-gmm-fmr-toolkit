"""Unit tests for the de-identification pipeline (synthetic data only)."""

import gzip
import json
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from lumbarmri.deidentify import (
    HEADER_TEXT_FIELDS,
    anonymize_filename,
    check_header_zero,
    deidentify,
    load_id_map,
    scrub_metadata,
    verify_dir,
    zero_header_text_fields,
)
from lumbarmri.io import read_image, write_image

ID_MAP = {"90000001": "case_01"}


def _write_dirty(path: Path, text: bytes, offset: int = 148, length: int = 80) -> None:
    """Plant ``text`` into a header text field (simulating legacy PHI)."""
    raw = bytearray(gzip.open(path, "rb").read())
    raw[offset : offset + length] = (text + b"\x00" * length)[:length]
    with path.open("wb") as fh:
        with gzip.GzipFile(filename="", mode="wb", fileobj=fh, mtime=0) as gz:
            gz.write(bytes(raw))


def test_anonymize_filename_rules():
    assert anonymize_filename("90000001_image.nii.gz", ID_MAP) == "case_01_image.nii.gz"
    assert anonymize_filename("90000001_ME_roi.nii.gz", ID_MAP) == "case_01_ME_label.nii.gz"
    assert anonymize_filename("90000001_P_roi.nii.gz", ID_MAP) == "case_01_P_label.nii.gz"
    with pytest.raises(ValueError):
        anonymize_filename("unmapped_image.nii.gz", ID_MAP)


def test_zero_header_text_fields(tmp_path):
    arr = np.arange(64, dtype=np.float32).reshape(1, 8, 8)
    path = tmp_path / "case_01_image.nii.gz"
    write_image(sitk.GetImageFromArray(arr), path)

    _write_dirty(path, b"patient 90000001")
    dirty = zero_header_text_fields(path)
    assert dirty["descrip"] is True

    ok, msg = check_header_zero([path])
    assert ok, msg

    # pixels untouched by the byte patch
    np.testing.assert_array_equal(sitk.GetArrayFromImage(read_image(path)), arr)


def test_all_five_fields_are_zeroed(tmp_path):
    arr = np.zeros((1, 4, 4), dtype=np.float32)
    path = tmp_path / "case_01_image.nii.gz"
    write_image(sitk.GetImageFromArray(arr), path)
    for name, offset, length in HEADER_TEXT_FIELDS:
        if name == "data_type":
            continue  # writer always leaves this zero; patch the others
        _write_dirty(path, b"x", offset=offset, length=length)
    dirty = zero_header_text_fields(path)
    assert dirty["db_name"] and dirty["aux_file"] and dirty["intent_name"]
    assert check_header_zero([path])[0]


def test_scrub_metadata_whitelist():
    img = sitk.GetImageFromArray(np.zeros((1, 4, 4), dtype=np.float32))
    img.SetMetaData("patient_id", "90000001")
    img.SetMetaData("institution", "some hospital")
    img.SetMetaData("descrip", "free text")
    img.SetMetaData("dim[1]", "4")

    scrub_metadata(img)
    keys = set(img.GetMetaDataKeys())
    assert "patient_id" not in keys
    assert "institution" not in keys
    assert "descrip" not in keys
    assert "dim[1]" in keys


def test_load_id_map_validation(tmp_path):
    good = tmp_path / "map.json"
    good.write_text(json.dumps({"90000001": "case_01"}), encoding="utf-8")
    assert load_id_map(good) == {"90000001": "case_01"}

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"90000001": "patient_one"}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_id_map(bad)


def test_deidentify_end_to_end(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    rng = np.random.RandomState(0)
    img_arr = rng.randint(0, 1000, size=(3, 6, 6)).astype(np.float32)
    write_image(sitk.GetImageFromArray(img_arr), src / "90000001_image.nii.gz")
    write_image(
        sitk.GetImageFromArray((img_arr > 500).astype(np.uint8)),
        src / "90000001_ME_roi.nii.gz",
    )
    _write_dirty(src / "90000001_image.nii.gz", b"legacy note")

    rows = deidentify(src, dst, ID_MAP)
    assert len(rows) == 2
    assert {r["file"] for r in rows} == {"case_01_image.nii.gz", "case_01_ME_label.nii.gz"}

    results = verify_dir(dst, exact_tokens=list(ID_MAP), src_dir=src, id_map=ID_MAP)
    failed = [(n, msg) for n, ok, msg in results if not ok]
    assert not failed, failed

    # pixel fidelity spot check
    np.testing.assert_array_equal(
        sitk.GetArrayFromImage(read_image(dst / "case_01_image.nii.gz")), img_arr
    )
