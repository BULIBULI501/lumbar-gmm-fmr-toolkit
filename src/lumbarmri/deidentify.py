"""NIfTI de-identification: header scrubbing, filename anonymisation, verification.

What this module does
---------------------
Three layers, applied to imaging files before they leave the study machine:

1. **Metadata dictionary scrubbing** -- every ``MetaDataDictionary`` key that
   is not on the technical whitelist (geometry / quantisation fields) is
   removed before writing.  Free-text keys such as ``descrip``, ``aux_file``,
   ``intent_name``, ``db_name``, ``ITK_FileNotes`` and anything matching a
   patient / institution pattern are always dropped.

2. **NIfTI-1 header text-field zeroing** -- five fixed byte fields of the
   348-byte NIfTI-1 header are forced to zero *after* writing, at the byte
   level, so the guarantee does not depend on writer behaviour:
   ``data_type`` (offset 4, 10 B), ``db_name`` (14, 18 B), ``descrip``
   (148, 80 B), ``aux_file`` (228, 24 B), ``intent_name`` (328, 16 B).
   Measured on the shipped study files these fields were *already* all zero;
   the byte patch is a safety net ("belt and braces").

3. **Filename anonymisation** -- real numeric IDs are replaced by ``case_NN``
   labels through a **local mapping file that is never committed**
   (``.gitignore`` excludes ``anonymization_map.*``).  Mask suffixes are
   unified to ``_label`` (the toolkit-wide naming convention).

Verification (:func:`verify_dir`, four checks)
----------------------------------------------
- **D1** every ``.nii.gz`` header text field is all zero;
- **D2** every file name matches ``case_NN_(image|ME_label|P_label).nii.gz``;
- **D3** no forbidden token anywhere: exact real IDs (from the local map)
  are byte-searched in *whole* files; structural patterns (``MRY``-style
  pseudonyms, the study table directory token, the CJK study word) are
  regex-scanned in text files and in imaging-file *header regions* only --
  scanning raw voxel bytes with short regexes would create false positives;
- **D4** pixel array MD5 + geometry (size/spacing/origin/direction) are
  identical before and after de-identification (compared per file pair).

.. warning::
   De-identification is **not** defacing.  Voxel content is preserved by
   design (D4).  Only metadata, headers and names are sanitised.  Do not use
   this module for head MRI where facial surface may be reconstructed.

Usage::

    python -m lumbarmri.deidentify run   --src SRC --dst examples/sample_data \\
                                         --map anonymization_map.json
    python -m lumbarmri.deidentify check --dir examples/sample_data \\
                                         [--src SRC] [--map anonymization_map.json]
    python -m lumbarmri.deidentify verify   # synthetic self-check
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import struct
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk

from .io import read_image as _read_sitk
from .io import write_image

__all__ = [
    "ANON_NAME_PATTERN",
    "HEADER_TEXT_FIELDS",
    "anonymize_filename",
    "check_forbidden",
    "check_header_zero",
    "check_names",
    "check_pixel_fidelity",
    "deidentify",
    "load_id_map",
    "pixel_signature",
    "scrub_metadata",
    "self_check",
    "verify_dir",
    "zero_header_text_fields",
]

# --------------------------------------------------------------------------
# NIfTI-1 header text fields (fixed byte offsets in the 348-byte header)
# --------------------------------------------------------------------------
HEADER_TEXT_FIELDS: Tuple[Tuple[str, int, int], ...] = (
    ("data_type", 4, 10),
    ("db_name", 14, 18),
    ("descrip", 148, 80),
    ("aux_file", 228, 24),
    ("intent_name", 328, 16),
)

_NIFTI1_MAGICS = (b"n+1\x00", b"ni1\x00")

# --------------------------------------------------------------------------
# Metadata whitelist: technical geometry / quantisation keys only
# --------------------------------------------------------------------------
_METADATA_WHITELIST = frozenset(
    {
        "datatype",
        "bitpix",
        "scl_slope",
        "scl_inter",
        "cal_min",
        "cal_max",
        "qform_code",
        "sform_code",
        "vox_offset",
        "xyzt_units",
        "dim_info",
        "toffset",
        "qfac",
        "nifti_type",
        "intent_code",
        "ITK_original_direction",
        "ITK_original_spacing",
        "ITK_original_origin",
        "ITK_sform_corrected",
    }
)

_METADATA_WHITELIST_PATTERNS = tuple(
    re.compile(p)
    for p in (
        r"dim\[\d\]",
        r"pixdim\[\d\]",
        r"srow_[xyz]",
        r"qoffset_[xyz]",
        r"quatern_[bcd]",
        r"slice_\w+",
        r"intent_p[123]",
    )
)

# Always dropped, regardless of whitelist.
_METADATA_DROP = frozenset(
    {"descrip", "aux_file", "intent_name", "db_name", "data_type", "ITK_FileNotes"}
)

# Patient / institution / free-text families (defence in depth).
_METADATA_RISKY = re.compile(
    r"patient|institution|birth|acces|physician|operator|subject|study|series|name|date",
    re.IGNORECASE,
)

# --------------------------------------------------------------------------
# Naming and scanning patterns
# --------------------------------------------------------------------------
ANON_NAME_PATTERN = re.compile(
    r"^(?P<case>case_\d{2})_(?P<part>image|ME_label|P_label)\.nii\.gz$"
)

_CASE_LABEL_PATTERN = re.compile(r"^case_\d{2}$")

# Structural forbidden patterns.  Assembled / escaped on purpose so that this
# source file itself never contains a matching literal (the scanner scans the
# whole repository, including this file).
_FORBIDDEN_REGEXPS: Tuple[re.Pattern, ...] = (
    re.compile(r"MRY\d+"),
    re.compile("clinic" + "_bic"),
    re.compile("\u8170\u690e"),  # the CJK study word, spelled via escapes
)

# Directories never scanned.
_SCAN_SKIP_DIRS = frozenset({".git", "__pycache__", ".pytest_cache", ".mypy_cache", "backup"})

# Imaging extensions whose payload is scanned in header region only.
_HEADER_SCAN_SUFFIXES = (".nii", ".nii.gz", ".nrrd", ".nrrd.gz", ".mha", ".mhd")

TOL_GEOMETRY = 1e-5


# --------------------------------------------------------------------------
# Mapping file
# --------------------------------------------------------------------------
def load_id_map(path: Path) -> Dict[str, str]:
    """Load the local ``{real_id: case_NN}`` JSON mapping file.

    The mapping file must stay on the study machine (``.gitignore`` already
    excludes ``anonymization_map.*``).  Values must match ``case_NN``.

    Args:
        path: JSON file mapping real IDs to case labels.

    Returns:
        ``{real_id: case_label}``.

    Raises:
        ValueError: mapping is malformed.
    """
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        raise ValueError(f"id map must be a non-empty JSON object: {path}")
    out: Dict[str, str] = {}
    for key, value in data.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"id map keys must be non-empty strings: {key!r}")
        if not isinstance(value, str) or not _CASE_LABEL_PATTERN.fullmatch(value):
            raise ValueError(
                f"id map values must match 'case_NN', got {value!r} for key {key!r}"
            )
        out[key] = value
    if len(set(out.values())) != len(out):
        raise ValueError("id map values (case labels) must be unique")
    return out


def anonymize_filename(name: str, id_map: Mapping[str, str]) -> str:
    """Rename ``{real_id}_ME_roi.nii.gz`` -> ``case_01_ME_label.nii.gz``.

    Rules: replace the real ID (longest key first) with its case label, then
    unify the mask suffix ``_roi`` -> ``_label``.

    Args:
        name: source file name (not a path).
        id_map: ``{real_id: case_label}``.

    Returns:
        The anonymised file name.

    Raises:
        ValueError: no mapped ID appears in ``name``.
    """
    for real_id, case in sorted(id_map.items(), key=lambda kv: -len(kv[0])):
        if real_id in name:
            new = name.replace(real_id, case, 1)
            return re.sub(r"_roi(?=\.nii\.gz$)", "_label", new)
    raise ValueError(f"file name does not contain any mapped id: {name}")


# --------------------------------------------------------------------------
# Layer 1: metadata dictionary scrubbing
# --------------------------------------------------------------------------
def _metadata_key_allowed(key: str) -> bool:
    if key in _METADATA_DROP:
        return False
    if _METADATA_RISKY.search(key):
        return False
    if key in _METADATA_WHITELIST:
        return True
    return any(p.fullmatch(key) for p in _METADATA_WHITELIST_PATTERNS)


def scrub_metadata(image: sitk.Image) -> sitk.Image:
    """Remove every non-whitelisted ``MetaDataDictionary`` key, in place.

    Args:
        image: image whose metadata should be scrubbed.

    Returns:
        The same image object (modified).
    """
    keep: Dict[str, str] = {}
    for key in list(image.GetMetaDataKeys()):
        if _metadata_key_allowed(key):
            keep[key] = image.GetMetaData(key)
        image.EraseMetaData(key)
    for key, value in keep.items():
        image.SetMetaData(key, value)
    return image


# --------------------------------------------------------------------------
# Layer 2: NIfTI-1 header text-field zeroing (byte level)
# --------------------------------------------------------------------------
def _read_container_bytes(path: Path) -> bytes:
    path = Path(path)
    if path.name.endswith(".gz"):
        with gzip.open(path, "rb") as fh:
            return fh.read()
    return path.read_bytes()


def _write_container_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    if path.name.endswith(".gz"):
        # Deterministic gzip: no embedded name, mtime=0.
        with path.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as fh:
                fh.write(data)
    else:
        path.write_bytes(data)


def zero_header_text_fields(path: Path) -> Dict[str, bool]:
    """Force the five NIfTI-1 header text fields to zero bytes.

    Operates on the raw container (gzip is decompressed and recompressed), so
    the result does not depend on the writer implementation.

    Args:
        path: ``.nii`` or ``.nii.gz`` file to patch in place.

    Returns:
        ``{field_name: was_dirty}`` -- whether the field held non-zero bytes
        before patching (an audit record for the de-identification log).

    Raises:
        ValueError: file is not a NIfTI-1 container.
    """
    path = Path(path)
    raw = bytearray(_read_container_bytes(path))
    if len(raw) < 352:
        raise ValueError(f"too short to be NIfTI-1: {path} ({len(raw)} bytes)")
    magic = bytes(raw[344:348])
    if magic not in _NIFTI1_MAGICS:
        raise ValueError(f"not a NIfTI-1 file (magic={magic!r}): {path}")
    dirty: Dict[str, bool] = {}
    for name, offset, length in HEADER_TEXT_FIELDS:
        segment = bytes(raw[offset : offset + length])
        dirty[name] = segment != b"\x00" * length
        raw[offset : offset + length] = b"\x00" * length
    _write_container_bytes(path, bytes(raw))
    return dirty


# --------------------------------------------------------------------------
# Pixel / geometry fingerprints (check D4)
# --------------------------------------------------------------------------
def pixel_signature(image: sitk.Image) -> Dict[str, object]:
    """MD5 of the voxel bytes plus the full geometry of an image.

    Args:
        image: image to fingerprint.

    Returns:
        ``{"dtype", "md5", "size", "spacing", "origin", "direction"}``.
    """
    arr = np.ascontiguousarray(sitk.GetArrayViewFromImage(image))
    return {
        "dtype": str(arr.dtype),
        "md5": hashlib.md5(arr.tobytes()).hexdigest(),
        "size": [int(v) for v in image.GetSize()],
        "spacing": [float(v) for v in image.GetSpacing()],
        "origin": [float(v) for v in image.GetOrigin()],
        "direction": [float(v) for v in image.GetDirection()],
    }


def _signatures_equal(a: Mapping[str, object], b: Mapping[str, object]) -> bool:
    if a["dtype"] != b["dtype"] or a["md5"] != b["md5"] or a["size"] != b["size"]:
        return False
    for key in ("spacing", "origin", "direction"):
        va, vb = list(a[key]), list(b[key])  # type: ignore[arg-type]
        if len(va) != len(vb):
            return False
        if any(abs(x - y) > TOL_GEOMETRY for x, y in zip(va, vb)):
            return False
    return True


# --------------------------------------------------------------------------
# De-identification pipeline
# --------------------------------------------------------------------------
def deidentify(
    src_dir: Path,
    dst_dir: Path,
    id_map: Mapping[str, str],
) -> List[Dict[str, object]]:
    """De-identify every ``{real_id}_*.nii.gz`` in ``src_dir`` into ``dst_dir``.

    Per file: read -> scrub metadata -> write under the anonymised name ->
    zero header text fields -> re-read and assert pixel/geometry fidelity.

    Args:
        src_dir: directory with the raw study files (read-only).
        dst_dir: output directory (created on demand).
        id_map: ``{real_id: case_label}``.

    Returns:
        One audit row per file (anonymous names only, safe to persist):
        ``{"file", "header_dirty", "metadata_dropped", "pixel_md5",
        "geometry_ok"}``.
    """
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    seen: set = set()
    for real_id, _case in id_map.items():
        for src in sorted(src_dir.glob(f"{real_id}_*.nii.gz")):
            if not src.is_file():
                continue
            # Skip when a longer mapped id is the actual prefix of this name.
            if any(o != real_id and len(o) > len(real_id) and src.name.startswith(o) for o in id_map):
                continue
            dst_name = anonymize_filename(src.name, id_map)
            if dst_name in seen:
                continue
            seen.add(dst_name)
            dst = dst_dir / dst_name
            image = _read_sitk(src)
            before = pixel_signature(image)
            dropped = sorted(k for k in image.GetMetaDataKeys() if not _metadata_key_allowed(k))
            scrub_metadata(image)
            write_image(image, dst)
            dirty = zero_header_text_fields(dst)
            after = pixel_signature(_read_sitk(dst))
            ok = _signatures_equal(before, after)
            if not ok:
                raise RuntimeError(f"pixel/geometry changed by de-identification: {src.name}")
            rows.append(
                {
                    "file": dst_name,
                    "header_dirty": {k: bool(v) for k, v in dirty.items()},
                    "metadata_dropped": dropped,
                    "pixel_md5": before["md5"],
                    "geometry_ok": ok,
                }
            )
    if not rows:
        raise FileNotFoundError(f"no files matched any mapped id in {src_dir}")
    return rows


# --------------------------------------------------------------------------
# Verification checks D1-D4
# --------------------------------------------------------------------------
def check_header_zero(paths: Iterable[Path]) -> Tuple[bool, str]:
    """D1: all header text fields of every NIfTI file are zero."""
    problems: List[str] = []
    n = 0
    for path in paths:
        path = Path(path)
        raw = _read_container_bytes(path)
        if len(raw) < 348:
            problems.append(f"{path.name}: too short")
            continue
        n += 1
        for name, offset, length in HEADER_TEXT_FIELDS:
            if raw[offset : offset + length] != b"\x00" * length:
                problems.append(f"{path.name}: non-zero header field '{name}'")
    if problems:
        return False, "; ".join(problems[:5])
    return True, f"{n} NIfTI header(s) clean"


def check_names(paths: Iterable[Path]) -> Tuple[bool, str]:
    """D2: every file name matches the anonymised naming pattern."""
    names = [Path(p).name for p in paths]
    problems = [n for n in names if not ANON_NAME_PATTERN.fullmatch(n)]
    if problems:
        return False, f"non-conforming names: {problems[:5]}"
    return True, f"{len(names)} name(s) conform"


def check_forbidden(
    root: Path,
    exact_tokens: Sequence[str] = (),
    regexps: Sequence[re.Pattern] = _FORBIDDEN_REGEXPS,
) -> Tuple[bool, str]:
    """D3: no forbidden token under ``root``.

    Exact tokens (real IDs) are byte-searched in whole file contents; the
    structural regexes are applied to text files and to imaging-file header
    regions only (voxel bytes would false-positive on short patterns).

    Args:
        root: directory to scan recursively (``.git`` etc. skipped).
        exact_tokens: literal strings that must not appear anywhere.
        regexps: structural patterns.

    Returns:
        ``(ok, message)``.
    """
    root = Path(root)
    hits: List[str] = []
    scanned = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in _SCAN_SKIP_DIRS for part in path.parts):
            continue
        rel = path.relative_to(root).as_posix()
        scanned += 1
        # File names are text.
        name_text = path.name
        for rx in regexps:
            if rx.search(name_text):
                hits.append(f"{rel}: name matches {rx.pattern!r}")
        for tok in exact_tokens:
            if tok in name_text:
                hits.append(f"{rel}: name contains exact token")
        try:
            payload = _read_container_bytes(path)
        except OSError as exc:  # unreadable file: report, do not hide
            hits.append(f"{rel}: unreadable ({exc})")
            continue
        is_imaging = path.name.endswith(_HEADER_SCAN_SUFFIXES)
        # Exact tokens: whole content for every file.
        for tok in exact_tokens:
            if tok.encode("utf-8") in payload:
                hits.append(f"{rel}: content contains exact token")
        # Structural regexes: text files fully, imaging files header region only.
        if is_imaging:
            header_len = 352
            if len(payload) >= 112:
                vox_offset = int(struct.unpack("<f", payload[108:112])[0])
                if 348 <= vox_offset <= len(payload):
                    header_len = vox_offset
            text = payload[:header_len].decode("latin-1", errors="ignore")
        else:
            text = payload.decode("utf-8", errors="ignore")
            if "\x00" in text[:4096] and not path.suffix in {".json", ".yaml", ".yml", ".csv", ".tsv"}:
                continue  # binary blob other than imaging: regex scan would noise
        for rx in regexps:
            if rx.search(text):
                hits.append(f"{rel}: content matches {rx.pattern!r}")
    if hits:
        return False, "; ".join(hits[:5])
    return True, f"{scanned} file(s) scanned, no forbidden token"


def check_pixel_fidelity(
    src_dir: Path,
    dst_dir: Path,
    id_map: Mapping[str, str],
) -> Tuple[bool, str]:
    """D4: pixel MD5 + geometry identical for every source/de-identified pair."""
    problems: List[str] = []
    n = 0
    for real_id in id_map:
        for src in sorted(Path(src_dir).glob(f"{real_id}_*.nii.gz")):
            if any(o != real_id and len(o) > len(real_id) and src.name.startswith(o) for o in id_map):
                continue
            dst = Path(dst_dir) / anonymize_filename(src.name, id_map)
            if not dst.is_file():
                problems.append(f"{dst.name}: missing in {dst_dir}")
                continue
            n += 1
            a = pixel_signature(_read_sitk(src))
            b = pixel_signature(_read_sitk(dst))
            if not _signatures_equal(a, b):
                problems.append(f"{dst.name}: pixel/geometry mismatch")
    if problems:
        return False, "; ".join(problems[:5])
    return True, f"{n} pair(s) pixel-identical"


def verify_dir(
    dst_dir: Path,
    exact_tokens: Sequence[str] = (),
    src_dir: Optional[Path] = None,
    id_map: Optional[Mapping[str, str]] = None,
) -> List[Tuple[str, bool, str]]:
    """Run the four de-identification checks on a directory.

    Args:
        dst_dir: directory holding the de-identified files.
        exact_tokens: real IDs to search for (from the local map).
        src_dir: raw source directory; enables D4 when given (with ``id_map``).
        id_map: mapping used for D4 pairing.

    Returns:
        ``[(name, ok, message), ...]`` record style.
    """
    dst_dir = Path(dst_dir)
    paths = sorted(p for p in dst_dir.rglob("*") if p.is_file() and p.name.endswith(".nii.gz"))
    records: List[Tuple[str, bool, str]] = []
    records.append(("D1: header text fields zero",) + check_header_zero(paths))
    records.append(("D2: anonymised file names",) + check_names(paths))
    records.append(("D3: no forbidden token",) + check_forbidden(dst_dir, exact_tokens))
    if src_dir is not None and id_map:
        records.append(("D4: pixel/geometry fidelity",) + check_pixel_fidelity(src_dir, dst_dir, id_map))
    else:
        records.append(("D4: pixel/geometry fidelity", True, "skipped (no --src/--map given)"))
    return records


# --------------------------------------------------------------------------
# Self-check on synthetic data (no real images involved)
# --------------------------------------------------------------------------
def self_check() -> List[Tuple[str, bool, str]]:
    """Synthetic end-to-end self-check of the de-identification pipeline.

    Builds tiny fake cases with dirty headers, risky metadata and real-style
    file names, then asserts the pipeline sanitises everything while leaving
    voxels untouched, and that each verification check detects its planted
    violation.  No study data is read.

    Returns:
        ``[(name, ok, message), ...]``.
    """
    import tempfile

    records: List[Tuple[str, bool, str]] = []

    def record(name: str, ok: bool, msg: str) -> None:
        records.append((name, bool(ok), msg))

    rng = np.random.RandomState(0)
    fake_id = "90000001"
    id_map = {fake_id: "case_01", "90000002": "case_02"}

    with tempfile.TemporaryDirectory(prefix="lumbarmri_deid_") as tmp:
        tmp_path = Path(tmp)
        src_dir, dst_dir = tmp_path / "src", tmp_path / "dst"
        src_dir.mkdir()

        # --- build two fake dirty cases -------------------------------
        for real_id in id_map:
            arr = rng.randint(0, 1000, size=(4, 8, 8)).astype(np.float32)
            img = sitk.GetImageFromArray(arr)
            img.SetSpacing((0.4, 0.4, 12.0))
            img.SetMetaData("patient_id", real_id)
            img.SetMetaData("institution", "some hospital")
            img.SetMetaData("descrip", f"scan of patient {real_id}")
            img.SetMetaData("dim[1]", "8")
            write_image(img, src_dir / f"{real_id}_image.nii.gz")
            mask = (arr > 500).astype(np.uint8)
            mimg = sitk.GetImageFromArray(mask)
            mimg.SetSpacing((0.4, 0.4, 12.0))
            write_image(mimg, src_dir / f"{real_id}_ME_roi.nii.gz")
            # dirty the header text fields at byte level (simulating legacy data)
            for field, offset, length in HEADER_TEXT_FIELDS:
                if field == "data_type":
                    continue
                _patch_text_field(src_dir / f"{real_id}_image.nii.gz", offset, length, real_id.encode())

        # S1: byte-level zeroing detects and clears injected text
        dirty_before = zero_header_text_fields(src_dir / f"{fake_id}_image.nii.gz")
        ok1 = dirty_before.get("descrip", False) and dirty_before.get("db_name", False)
        clean1, msg1 = check_header_zero([src_dir / f"{fake_id}_image.nii.gz"])
        record("S1: header text fields zeroed", ok1 and clean1, msg1)

        # S2: metadata scrubbing keeps whitelist only
        probe = _read_sitk(src_dir / f"{fake_id}_image.nii.gz")
        probe.SetMetaData("patient_id", fake_id)
        probe.SetMetaData("dim[1]", "8")
        scrub_metadata(probe)
        keys = set(probe.GetMetaDataKeys())
        ok2 = "patient_id" not in keys and "descrip" not in keys and "dim[1]" in keys
        record("S2: metadata whitelist scrub", ok2, f"kept keys: {sorted(keys)}")

        # S3: filename rules
        names = {
            f"{fake_id}_image.nii.gz": "case_01_image.nii.gz",
            f"{fake_id}_ME_roi.nii.gz": "case_01_ME_label.nii.gz",
            f"{fake_id}_P_roi.nii.gz": "case_01_P_label.nii.gz",
        }
        ok3 = all(anonymize_filename(k, id_map) == v for k, v in names.items())
        record("S3: filename anonymisation rules", ok3, str(names))

        # S4: full pipeline + four checks
        rows = deidentify(src_dir, dst_dir, id_map)
        tokens = list(id_map.keys())
        results = verify_dir(dst_dir, exact_tokens=tokens, src_dir=src_dir, id_map=id_map)
        ok4 = len(rows) == 4 and all(r[1] for r in results)
        record("S4: end-to-end pipeline + D1-D4", ok4, "; ".join(f"{n}={'PASS' if o else 'FAIL'}" for n, o, _ in results))

        # S5a: planted dirty header is detected
        victim = sorted(dst_dir.glob("*.nii.gz"))[0]
        _patch_text_field(victim, 148, 80, b"planted-phi")
        ok5a, _ = check_header_zero([victim])
        record("S5a: D1 detects planted header text", not ok5a, "planted descrip detected" if not ok5a else "MISSED")
        zero_header_text_fields(victim)

        # S5b: planted forbidden token in a text file is detected
        leak = dst_dir / "notes.txt"
        leak.write_text("study word \u8170\u690e and id " + fake_id, encoding="utf-8")
        ok5b, _ = check_forbidden(dst_dir, exact_tokens=tokens)
        record("S5b: D3 detects planted token", not ok5b, "planted token detected" if not ok5b else "MISSED")
        leak.unlink()

        # S5c: non-conforming file name is detected
        bad = dst_dir / "totally_wrong_name.nii.gz"
        bad.write_bytes(b"\x00" * 352)
        ok5c, _ = check_names([bad])
        record("S5c: D2 detects bad file name", not ok5c, "bad name detected" if not ok5c else "MISSED")
        bad.unlink()

        # S6: id map validation rejects bad case labels
        bad_map = tmp_path / "bad_map.json"
        bad_map.write_text(json.dumps({fake_id: "patient_one"}), encoding="utf-8")
        try:
            load_id_map(bad_map)
            ok6 = False
            msg6 = "bad map accepted"
        except ValueError:
            ok6 = True
            msg6 = "bad case label rejected"
        record("S6: id map validation", ok6, msg6)

    return records


def _patch_text_field(path: Path, offset: int, length: int, text: bytes) -> None:
    """Plant ``text`` into a header text field (test helper)."""
    raw = bytearray(_read_container_bytes(path))
    raw[offset : offset + length] = (text + b"\x00" * length)[:length]
    _write_container_bytes(path, bytes(raw))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _cmd_run(args: argparse.Namespace) -> int:
    id_map = load_id_map(args.map)
    rows = deidentify(args.src, args.dst, id_map)
    manifest = Path(args.dst) / "deidentify_manifest.json"
    manifest.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    for row in rows:
        print(f"OK {row['file']}  md5={row['pixel_md5'][:12]}  header_dirty={row['header_dirty']}")
    print(f"wrote {manifest} ({len(rows)} files)")
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    tokens: List[str] = []
    id_map = None
    if args.map:
        id_map = load_id_map(args.map)
        tokens = list(id_map.keys())
    tokens.extend(args.forbidden or [])
    records = verify_dir(
        args.dir,
        exact_tokens=tokens,
        src_dir=args.src,
        id_map=id_map,
    )
    width = max(len(n) for n, _, _ in records)
    for name, ok, msg in records:
        print(f"{name:<{width}}  {'PASS' if ok else 'FAIL'}  {msg}")
    print(f"Overall: {'PASS' if all(r[1] for r in records) else 'FAIL'}")
    return 0 if all(r[1] for r in records) else 1


def _cmd_verify(_args: argparse.Namespace) -> int:
    records = self_check()
    width = max(len(n) for n, _, _ in records)
    for name, ok, msg in records:
        print(f"{name:<{width}}  {'PASS' if ok else 'FAIL'}  {msg}")
    print(f"Overall: {'PASS' if all(r[1] for r in records) else 'FAIL'}")
    return 0 if all(r[1] for r in records) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lumbarmri.deidentify", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="de-identify a directory of study files")
    p_run.add_argument("--src", type=Path, required=True)
    p_run.add_argument("--dst", type=Path, required=True)
    p_run.add_argument("--map", type=Path, required=True, help="local id->case JSON (never commit)")
    p_run.set_defaults(func=_cmd_run)

    p_check = sub.add_parser("check", help="run the four de-identification checks")
    p_check.add_argument("--dir", type=Path, required=True)
    p_check.add_argument("--src", type=Path, default=None, help="enables D4 when given")
    p_check.add_argument("--map", type=Path, default=None, help="local id->case JSON")
    p_check.add_argument("--forbidden", nargs="*", default=[], help="extra exact tokens")
    p_check.set_defaults(func=_cmd_check)

    p_verify = sub.add_parser("verify", help="synthetic self-check (no real data)")
    p_verify.set_defaults(func=_cmd_verify)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
