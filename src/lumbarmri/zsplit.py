"""z-axis splitting / continuity relabeling / label-group export for ROI masks.

Background
----------
Lumbar paraspinal ROI masks (``*_label.nii.gz``) often need three kinds of
z-axis (head-foot) post-processing in the research pipeline:

1. **split**: slice the whole mask into single-slice files (each stored as a
   ``(1, y, x)`` 3-D mask, preserving spacing/origin/direction; the origin's
   z component is offset by ``i * spacing[2]`` per slice) for per-slice
   imaging/radiomics feature extraction;
2. **relabel**: detect z-continuous runs of the target label and relabel them
   as 1, 2, 3, ... (runs separated by empty slices) for per-vertebra/segment
   structure studies;
3. **export**: export each continuous label group as a full-size binary mask
   (original dimensions preserved) for per-segment statistics.

Naming rules (compatible with the historical pipeline)
------------------------------------------------------
split/export output names come from :func:`split_output_name`: strip the
trailing ``_label{draw}`` to get the base, then name outputs
``{base}{draw_token}_{index}{suffix}``:

- ``case_01_ME_label.nii.gz``   -> ``case_01_ME_1_label.nii.gz``
  (byte-compatible with the legacy split_mask_by_z.py);
- ``case_01_ME_label2.nii.gz``  -> ``case_01_ME_label2_1_label.nii.gz``
  (keeps the repeat-draw marker, avoids colliding with first-draw outputs);
- per-vertebra masks ``case_01_P_1_label.nii.gz`` -> ``case_01_P_1_1_label.nii.gz``
  (never overwrites the input file itself).

relabel outputs are ``{mask_stem}{suffix}.nii.gz`` (default suffix
``_relabeled``) -- an appended name, so it never collides with the input.

Data compatibility
------------------
- Mask discovery reuses the ``{stem}_label{draw}.nii.gz`` pattern: whole
  segment, per-vertebra and repeat-draw (``_label2``) naming are all picked
  up; historical outputs (``*_label_new.nii.gz``, ``*_relabeled.nii.gz``) do
  not match and are excluded.
- ``--target-label auto`` (default) takes the unique non-zero value, covering
  both {0,1} (first draw) and {0,2} (repeat draw) masks; when the non-zero
  value is not unique (e.g. a previously relabeled multi-label mask) it must
  be given explicitly.
- Non-ASCII paths go through the ``chdir`` + ASCII-filename channel (the
  documented SimpleITK ENOENT workaround, see :mod:`lumbarmri.io`).

Usage
-----
    lumbarmri zsplit verify                          # synthetic self-check
    lumbarmri zsplit split   --mask case_01_ME_label.nii.gz
    lumbarmri zsplit split   --root ./cases [--recursive]
    lumbarmri zsplit relabel --mask case_01_ME_label.nii.gz [--target-label auto]
    lumbarmri zsplit export  --mask case_01_ME_label.nii.gz

Safety boundary: source masks are read-only; outputs default to the mask's
own directory (``--output-dir`` overrides); existing outputs are never
overwritten unless ``--overwrite`` is given, and all output paths are
pre-checked before writing so no partial output is produced.  Note that split
and export share the default suffix (``_label.nii.gz``) and the same index
rule: running both on the same mask with default suffixes makes the second
run fail on existing outputs -- this is intentional anti-overwrite behaviour;
use ``--output-suffix`` to distinguish them.

Import example::

    from lumbarmri.zsplit import split_mask_along_z
    result = split_mask_along_z(Path("case_01_ME_label.nii.gz"))
    print(len(result.output_files))
"""

from __future__ import annotations

import argparse
import logging
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import SimpleITK as sitk

from .io import chdir

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: split/export default output suffix (legacy-compatible)
DEFAULT_SPLIT_SUFFIX = "_label.nii.gz"

#: relabel default output suffix (appended name, never equals the input name)
DEFAULT_RELABEL_SUFFIX = "_relabeled"

#: Mask discovery pattern: {any stem}_label{draw}.nii.gz
#: Covers whole-segment (case_01_ME_label.nii.gz), per-vertebra
#: (case_01_P_1_label.nii.gz) and repeat-draw (case_01_ME_label2.nii.gz)
#: naming; historical outputs *_label_new.nii.gz / *_relabeled.nii.gz do not
#: match and are therefore excluded.
MASK_NAME_PATTERN = re.compile(r"^(?P<stem>.+)_label(?P<draw>\d*)\.nii\.gz$")

#: Synthetic self-check parameters: two non-contiguous runs z=0..2 and z=5..7
#: (z=3..4 empty)
SELF_CHECK_SHAPE = (8, 4, 4)
SELF_CHECK_SPACING = (2.0, 0.5, 0.5)
SELF_CHECK_ORIGIN = (10.0, -3.0, 7.5)

#: Floating-point comparison tolerances (tiered, shared across the toolkit)
TOL_EXACT = 1e-9
TOL_FLOAT = 1e-3


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ZGroup:
    """One z-continuous run of slices (endpoints included)."""

    start_z: int  # first slice (inclusive)
    end_z: int  # last slice (inclusive)

    @property
    def n_slices(self) -> int:
        """Number of slices in the run."""
        return self.end_z - self.start_z + 1


@dataclass
class MaskInfo:
    """Full mask information (array + spatial metadata)."""

    path: Path
    array: np.ndarray
    spacing: Tuple[float, float, float]
    origin: Tuple[float, float, float]
    direction: Tuple[float, ...]


@dataclass
class SplitResult:
    """split result: per-slice single-slice files."""

    mask_path: Path
    output_files: List[Path]


@dataclass
class RelabelResult:
    """relabel result: z-continuity runs relabeled."""

    mask_path: Path
    output_path: Path
    groups: List[ZGroup]
    target_label: int


@dataclass
class ExportResult:
    """export result: full-size binary mask per continuous label group."""

    mask_path: Path
    output_files: List[Path]
    groups: List[ZGroup]
    target_label: int


@dataclass
class BatchResult:
    """Batch summary (per-file failures do not abort; exit code follows)."""

    ok: List[Path] = field(default_factory=list)
    failed: List[Tuple[Path, str]] = field(default_factory=list)
    outputs: List[Path] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logging(log_dir: Optional[Path], console_level: int = logging.INFO) -> logging.Logger:
    """Configure dual-channel logging (file + console).

    Args:
        log_dir: directory for the log file; None = console only.
        console_level: console logging level.

    Returns:
        The configured logger.
    """
    logger = logging.getLogger("zsplit")
    logger.setLevel(logging.DEBUG)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"zsplit_{datetime.now():%Y%m%d_%H%M%S}.log"
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(console_level)
    ch.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    logger.addHandler(ch)
    return logger


# ---------------------------------------------------------------------------
# Mask I/O (chdir-safe; see lumbarmri.io)
# ---------------------------------------------------------------------------


def read_mask_info(mask_path: Path) -> MaskInfo:
    """Read a mask plus its spatial metadata (spacing/origin/direction).

    Args:
        mask_path: mask file path (``.nii.gz``).

    Returns:
        MaskInfo; the SimpleITK array order is (z, y, x).

    Raises:
        FileNotFoundError: mask file missing.
        ValueError: mask is not a 3-D array.
    """
    mask_path = Path(mask_path)
    if not mask_path.is_file():
        raise FileNotFoundError(f"mask file not found: {mask_path}")

    with chdir(mask_path.parent):
        image = sitk.ReadImage(mask_path.name)
        array = sitk.GetArrayFromImage(image)

    if array.ndim != 3:
        raise ValueError(
            f"mask must be a 3-D array (z, y, x), got shape {array.shape}: {mask_path}"
        )

    return MaskInfo(
        path=mask_path,
        array=array,
        spacing=tuple(float(s) for s in image.GetSpacing()),
        origin=tuple(float(o) for o in image.GetOrigin()),
        direction=tuple(float(d) for d in image.GetDirection()),
    )


def _save_with_metadata(
    array: np.ndarray,
    info: MaskInfo,
    out_dir: Path,
    out_name: str,
    origin: Optional[Tuple[float, float, float]] = None,
) -> Path:
    """Save an array as a mask file, preserving spacing/direction/origin.

    Args:
        array: array to save.
        info: reference mask info (provides spacing/direction; origin can be
            overridden).
        out_dir: output directory.
        out_name: output filename (ASCII).
        origin: origin override (e.g. per-slice z offset for split);
            None keeps info.origin.

    Returns:
        Output file path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(info.spacing)
    image.SetDirection(info.direction)
    image.SetOrigin(origin if origin is not None else info.origin)
    with chdir(out_dir):
        sitk.WriteImage(image, out_name)
    return out_dir / out_name


def _ensure_not_exists(path: Path, overwrite: bool) -> None:
    """Output collision guard: raise when the target exists and overwrite is off.

    Args:
        path: target output path.
        overwrite: whether overwriting is allowed.

    Raises:
        FileExistsError: path exists and overwrite=False.
    """
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"output already exists (overwrite refused by default, use --overwrite): {path}"
        )


# ---------------------------------------------------------------------------
# Naming rules (legacy-compatible + collision-proof)
# ---------------------------------------------------------------------------


def strip_label_stem(mask_name: str) -> Tuple[str, str]:
    """Strip the trailing ``_label{draw}`` from a mask name -> (base, draw).

    Args:
        mask_name: mask filename, e.g. ``case_01_ME_label2.nii.gz``.

    Returns:
        (base, draw): base has the trailing ``_label{draw}`` removed; draw is
        the annotation-round token ('' = first draw, '2' = repeat draw).

    Raises:
        ValueError: name does not match the ``*_label.nii.gz`` pattern.
    """
    m = MASK_NAME_PATTERN.match(mask_name)
    if not m:
        raise ValueError(f"mask name does not match the *_label.nii.gz pattern: {mask_name}")
    return m.group("stem"), m.group("draw")


def split_output_name(mask_name: str, index: int, output_suffix: str) -> str:
    """Build the filename of the index-th split/export output.

    Rule: ``{base}{draw_token}_{index}{output_suffix}``, where base is the
    mask name minus the trailing ``_label{draw}``, and draw_token is
    ``_label{draw}`` when draw is non-empty (keeps the repeat-draw marker,
    collision-proof).

    Args:
        mask_name: mask filename.
        index: 1-based index.
        output_suffix: output suffix including extension
            (default ``_label.nii.gz``).

    Returns:
        Output filename.

    Raises:
        ValueError: mask_name does not match the naming pattern.

    Examples:
        >>> split_output_name("case_01_ME_label.nii.gz", 1, "_label.nii.gz")
        'case_01_ME_1_label.nii.gz'
        >>> split_output_name("case_01_ME_label2.nii.gz", 1, "_label.nii.gz")
        'case_01_ME_label2_1_label.nii.gz'
    """
    base, draw = strip_label_stem(mask_name)
    draw_token = f"_label{draw}" if draw else ""
    return f"{base}{draw_token}_{index}{output_suffix}"


# ---------------------------------------------------------------------------
# z-continuity detection and target-label resolution
# ---------------------------------------------------------------------------


def find_z_groups(array: np.ndarray, target_label: int) -> List[ZGroup]:
    """Detect z-continuous runs containing the target label.

    Same per-slice semantics as the legacy relabel/export: a slice belongs to
    a run when it contains at least one target-label voxel; empty slices break
    runs.

    Args:
        array: mask array (z, y, x).
        target_label: target label value.

    Returns:
        ZGroup list in ascending z order (empty when no target voxels).
    """
    has_label = np.any(array == target_label, axis=(1, 2)).tolist()
    groups: List[ZGroup] = []
    start: Optional[int] = None
    for z, flag in enumerate(has_label):
        if flag and start is None:
            start = z
        elif not flag and start is not None:
            groups.append(ZGroup(start, z - 1))
            start = None
    if start is not None:
        groups.append(ZGroup(start, len(has_label) - 1))
    return groups


def resolve_target_label(array: np.ndarray, target_label: Union[int, str] = "auto") -> int:
    """Resolve the target label: ``"auto"`` takes the unique non-zero value.

    auto mode covers {0,1} (first draw) and {0,2} (repeat draw) masks; when
    the non-zero value is not unique the label must be explicit.

    Args:
        array: mask array.
        target_label: ``"auto"`` or an integer value (e.g. 1/2).

    Returns:
        Target label as int.

    Raises:
        ValueError: auto mode with an all-zero mask or multiple non-zero
            values; or a string that is neither "auto" nor an integer.
    """
    if isinstance(target_label, str):
        if target_label.strip().lower() != "auto":
            raise ValueError(f'target_label must be "auto" or an int, got: {target_label!r}')
        nonzero = np.unique(array)
        nonzero = nonzero[nonzero != 0]
        if nonzero.size == 0:
            raise ValueError("mask has no non-zero label; nothing to process")
        if nonzero.size > 1:
            raise ValueError(
                f"mask has multiple non-zero values {nonzero.tolist()}; "
                "specify --target-label explicitly (e.g. a previously relabeled multi-label mask)"
            )
        return int(nonzero[0])
    return int(target_label)


def parse_target_label_arg(value: Union[int, str]) -> Union[int, str]:
    """Parse the CLI ``--target-label`` argument: 'auto' or an integer string.

    Args:
        value: raw argument value.

    Returns:
        ``"auto"`` or an int.

    Raises:
        ValueError: neither 'auto' nor parseable as an integer.
    """
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if text.lower() == "auto":
        return "auto"
    try:
        return int(text)
    except ValueError as e:
        raise ValueError(f"--target-label must be 'auto' or an integer, got: {value!r}") from e


# ---------------------------------------------------------------------------
# Core feature 1: per-slice z split
# ---------------------------------------------------------------------------


def split_mask_along_z(
    mask_path: Path,
    output_dir: Optional[Path] = None,
    output_suffix: str = DEFAULT_SPLIT_SUFFIX,
    overwrite: bool = False,
    logger: Optional[logging.Logger] = None,
) -> SplitResult:
    """Slice a whole mask along z into single-slice files.

    Each slice is stored as a ``(1, y, x)`` 3-D mask preserving
    spacing/direction; the origin's z component is offset by
    ``i * spacing[2]`` per slice so each slice keeps its true spatial
    position.

    Args:
        mask_path: mask file path (``.nii.gz``).
        output_dir: output directory; None = the mask's own directory.
        output_suffix: output suffix (default ``_label.nii.gz``).
        overwrite: allow overwriting existing outputs (refused by default).
        logger: optional logger.

    Returns:
        SplitResult (outputs sorted by ascending z).

    Raises:
        FileNotFoundError: mask missing.
        FileExistsError: output exists and overwrite=False (all paths are
            pre-checked before writing, so no partial output is produced).
        ValueError: mask not 3-D or name does not match the pattern.
    """
    info = read_mask_info(mask_path)
    out_dir = Path(output_dir) if output_dir else info.path.parent
    n_z = info.array.shape[0]

    # Pre-check every output path before writing anything (no partial output).
    outputs = [
        out_dir / split_output_name(info.path.name, i + 1, output_suffix)
        for i in range(n_z)
    ]
    for out_path in outputs:
        _ensure_not_exists(out_path, overwrite)

    written: List[Path] = []
    for i, out_path in enumerate(outputs):
        slice_array = info.array[i:i + 1, :, :]
        origin = (info.origin[0], info.origin[1], info.origin[2] + i * info.spacing[2])
        written.append(_save_with_metadata(slice_array, info, out_dir, out_path.name, origin=origin))

    if logger:
        logger.info(
            f"{info.path.name}: split along z into {len(written)} single-slice mask(s) -> {out_dir}"
        )
    return SplitResult(mask_path=info.path, output_files=written)


# ---------------------------------------------------------------------------
# Core feature 2: z-continuity relabeling
# ---------------------------------------------------------------------------


def relabel_z_continuity(
    mask_path: Path,
    output_dir: Optional[Path] = None,
    output_suffix: str = DEFAULT_RELABEL_SUFFIX,
    target_label: Union[int, str] = "auto",
    overwrite: bool = False,
    logger: Optional[logging.Logger] = None,
) -> RelabelResult:
    """Relabel target-label voxels by z-continuity: run 1 -> 1, run 2 -> 2, ...

    Legacy semantics preserved: only target-label voxels are relabeled, all
    other voxels keep their value; the output array is promoted to int32 to
    hold run indices.  Output name is ``{mask_stem}{output_suffix}.nii.gz``.

    Args:
        mask_path: mask file path.
        output_dir: output directory; None = the mask's own directory.
        output_suffix: output suffix (default ``_relabeled``, appended name,
            collision-proof).
        target_label: target label; ``"auto"`` takes the unique non-zero
            value (default).
        overwrite: allow overwriting existing outputs (refused by default).
        logger: optional logger.

    Returns:
        RelabelResult (runs + resolved target label).

    Raises:
        FileNotFoundError: mask missing.
        FileExistsError: output exists and overwrite=False.
        ValueError: target label unresolvable (all-zero / multiple values).
    """
    info = read_mask_info(mask_path)
    target = resolve_target_label(info.array, target_label)
    groups = find_z_groups(info.array, target)

    out_array = info.array.astype(np.int32, copy=True)
    for group_idx, group in enumerate(groups, start=1):
        for z in range(group.start_z, group.end_z + 1):
            out_array[z, :, :] = np.where(info.array[z, :, :] == target, group_idx, out_array[z, :, :])

    out_dir = Path(output_dir) if output_dir else info.path.parent
    out_name = f"{info.path.name[:-len('.nii.gz')]}{output_suffix}.nii.gz"
    out_path = out_dir / out_name
    _ensure_not_exists(out_path, overwrite)
    _save_with_metadata(out_array, info, out_dir, out_name)

    if logger:
        spans = ", ".join(f"group{i}(z={g.start_z}~{g.end_z})" for i, g in enumerate(groups, 1))
        logger.info(
            f"{info.path.name}: target={target} found {len(groups)} continuous run(s) "
            f"({spans}) -> {out_name}"
        )
    return RelabelResult(mask_path=info.path, output_path=out_path, groups=groups, target_label=target)


# ---------------------------------------------------------------------------
# Core feature 3: label-group export as full-size binary masks
# ---------------------------------------------------------------------------


def export_z_groups(
    mask_path: Path,
    output_dir: Optional[Path] = None,
    output_suffix: str = DEFAULT_SPLIT_SUFFIX,
    target_label: Union[int, str] = "auto",
    overwrite: bool = False,
    logger: Optional[logging.Logger] = None,
) -> ExportResult:
    """Export each continuous label group as a full-size binary mask.

    The group_idx-th output holds 1 for target-label voxels inside that
    group's z range and 0 elsewhere (layers outside the group are all 0).
    Naming matches split (:func:`split_output_name`).

    Args:
        mask_path: mask file path.
        output_dir: output directory; None = the mask's own directory.
        output_suffix: output suffix (default ``_label.nii.gz``).
        target_label: target label; ``"auto"`` takes the unique non-zero
            value (default).
        overwrite: allow overwriting existing outputs (refused by default).
        logger: optional logger.

    Returns:
        ExportResult (outputs sorted by group index).

    Raises:
        FileNotFoundError: mask missing.
        FileExistsError: output exists and overwrite=False (pre-checked).
        ValueError: target label unresolvable (all-zero / multiple values).
    """
    info = read_mask_info(mask_path)
    target = resolve_target_label(info.array, target_label)
    groups = find_z_groups(info.array, target)

    out_dir = Path(output_dir) if output_dir else info.path.parent
    outputs = [
        out_dir / split_output_name(info.path.name, idx, output_suffix)
        for idx, _ in enumerate(groups, start=1)
    ]
    for out_path in outputs:
        _ensure_not_exists(out_path, overwrite)

    written: List[Path] = []
    for group_idx, (group, out_path) in enumerate(zip(groups, outputs), start=1):
        group_array = np.zeros(info.array.shape, dtype=np.int32)
        for z in range(group.start_z, group.end_z + 1):
            group_array[z, :, :] = np.where(info.array[z, :, :] == target, 1, 0)
        written.append(_save_with_metadata(group_array, info, out_dir, out_path.name))

    if logger:
        logger.info(
            f"{info.path.name}: target={target} exported {len(written)} group binary mask(s) "
            f"-> {out_dir}"
        )
    return ExportResult(
        mask_path=info.path, output_files=written, groups=groups, target_label=target
    )


# ---------------------------------------------------------------------------
# Batch processing (auto-discovery; per-file failures do not abort)
# ---------------------------------------------------------------------------


def discover_masks(root_dir: Path, recursive: bool = False,
                   case_list: Optional[Sequence[str]] = None) -> List[Path]:
    """Enumerate mask files to process under root_dir.

    - ``recursive=False``: flat masks in root_dir + masks in direct
      subdirectories;
    - ``recursive=True``: every level (root included);
    - ``case_list``: keep only masks under these directories (batched runs).

    Historical outputs (``*_label_new.nii.gz``, ``*_relabeled.nii.gz``) do
    not match the discovery pattern and are excluded.

    Args:
        root_dir: batch root directory.
        recursive: whether to recurse.
        case_list: directory-name whitelist; None = no filtering.

    Returns:
        Sorted list of mask file paths.

    Raises:
        FileNotFoundError: root_dir is not a directory.
    """
    root_dir = Path(root_dir)
    if not root_dir.is_dir():
        raise FileNotFoundError(f"root directory not found: {root_dir}")

    wanted = set(case_list) if case_list is not None else None

    def keep(p: Path) -> bool:
        if wanted is None:
            return True
        d = p.parent.name
        return d in wanted or d.removeprefix("STU_") in wanted

    paths = {p for p in root_dir.glob("*.nii.gz")
             if MASK_NAME_PATTERN.match(p.name) and keep(p)}
    if recursive:
        for p in root_dir.rglob("*.nii.gz"):
            if MASK_NAME_PATTERN.match(p.name) and keep(p):
                paths.add(p)
    else:
        for child in sorted(root_dir.iterdir()):
            if child.is_dir():
                for p in child.glob("*.nii.gz"):
                    if MASK_NAME_PATTERN.match(p.name) and keep(p):
                        paths.add(p)
    return sorted(paths)


def _run_batch(
    root_dir: Path,
    single_fn,
    logger: Optional[logging.Logger],
    recursive: bool,
    output_dir: Optional[Path],
    output_suffix: str,
    target_label: Union[int, str],
    overwrite: bool,
    case_list: Optional[Sequence[str]] = None,
) -> BatchResult:
    """Run one single-file processor over a batch (internal helper)."""
    batch = BatchResult()
    for mask_path in discover_masks(root_dir, recursive, case_list):
        try:
            if single_fn is split_mask_along_z:
                res = single_fn(mask_path, output_dir, output_suffix, overwrite, logger)
                batch.outputs.extend(res.output_files)
            elif single_fn is export_z_groups:
                res = single_fn(mask_path, output_dir, output_suffix, target_label, overwrite, logger)
                batch.outputs.extend(res.output_files)
            else:  # relabel_z_continuity: single output file
                res = single_fn(mask_path, output_dir, output_suffix, target_label, overwrite, logger)
                batch.outputs.append(res.output_path)
            batch.ok.append(mask_path)
        except Exception as e:  # per-file failure does not abort the batch
            batch.failed.append((mask_path, str(e)))
            if logger:
                logger.error(f"failed: {mask_path}: {e}")
    return batch


def split_tree(
    root_dir: Path,
    output_dir: Optional[Path] = None,
    output_suffix: str = DEFAULT_SPLIT_SUFFIX,
    recursive: bool = False,
    overwrite: bool = False,
    logger: Optional[logging.Logger] = None,
    case_list: Optional[Sequence[str]] = None,
) -> BatchResult:
    """Batch-split every discovered mask under root_dir along z.

    Args:
        root_dir: batch root directory.
        output_dir: output directory; None = each mask's own directory.
        output_suffix: output suffix.
        recursive: recurse into subdirectories.
        overwrite: allow overwriting existing outputs.
        logger: optional logger.
        case_list: process only masks under these directories (batched runs).

    Returns:
        BatchResult (ok/failed/outputs).

    Raises:
        FileNotFoundError: root_dir is not a directory.
    """
    return _run_batch(root_dir, split_mask_along_z, logger, recursive,
                      output_dir, output_suffix, "auto", overwrite, case_list)


def relabel_tree(
    root_dir: Path,
    output_dir: Optional[Path] = None,
    output_suffix: str = DEFAULT_RELABEL_SUFFIX,
    target_label: Union[int, str] = "auto",
    recursive: bool = False,
    overwrite: bool = False,
    logger: Optional[logging.Logger] = None,
    case_list: Optional[Sequence[str]] = None,
) -> BatchResult:
    """Batch-relabel every discovered mask under root_dir by z-continuity.

    Args:
        root_dir: batch root directory.
        output_dir: output directory; None = each mask's own directory.
        output_suffix: output suffix.
        target_label: target label; ``"auto"`` takes the unique non-zero value.
        recursive: recurse into subdirectories.
        overwrite: allow overwriting existing outputs.
        logger: optional logger.
        case_list: process only masks under these directories (batched runs).

    Returns:
        BatchResult.

    Raises:
        FileNotFoundError: root_dir is not a directory.
    """
    return _run_batch(root_dir, relabel_z_continuity, logger, recursive,
                      output_dir, output_suffix, target_label, overwrite, case_list)


def export_tree(
    root_dir: Path,
    output_dir: Optional[Path] = None,
    output_suffix: str = DEFAULT_SPLIT_SUFFIX,
    target_label: Union[int, str] = "auto",
    recursive: bool = False,
    overwrite: bool = False,
    logger: Optional[logging.Logger] = None,
    case_list: Optional[Sequence[str]] = None,
) -> BatchResult:
    """Batch-export continuous label groups of every discovered mask.

    Args:
        root_dir: batch root directory.
        output_dir: output directory; None = each mask's own directory.
        output_suffix: output suffix.
        target_label: target label; ``"auto"`` takes the unique non-zero value.
        recursive: recurse into subdirectories.
        overwrite: allow overwriting existing outputs.
        logger: optional logger.
        case_list: process only masks under these directories (batched runs).

    Returns:
        BatchResult.

    Raises:
        FileNotFoundError: root_dir is not a directory.
    """
    return _run_batch(root_dir, export_z_groups, logger, recursive,
                      output_dir, output_suffix, target_label, overwrite, case_list)


# ---------------------------------------------------------------------------
# Self-check (synthetic data; no external dataset needed)
# ---------------------------------------------------------------------------


def self_check() -> List[Tuple[str, bool, str]]:
    """z-processing self-check on a synthetic (8,4,4) mask.

    Synthetic data: two non-contiguous runs z=0..2 and z=5..7 (z=3..4 empty),
    spacing (2.0, 0.5, 0.5), origin (10.0, -3.0, 7.5).

    Returns:
        [(check name, passed, detail)] list.
    """
    results: List[Tuple[str, bool, str]] = []

    def record(name: str, ok: bool, msg: str) -> None:
        results.append((name, bool(ok), msg))

    shape = SELF_CHECK_SHAPE
    spacing = SELF_CHECK_SPACING
    origin = SELF_CHECK_ORIGIN
    direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    def close(a: Sequence[float], b: Sequence[float]) -> bool:
        return len(a) == len(b) and all(abs(x - y) < TOL_EXACT for x, y in zip(a, b))

    # {0,1} (first draw) and {0,2} (repeat draw) masks with identical geometry
    mask01 = np.zeros(shape, dtype=np.int32)
    mask01[0:3, 0, 0:2] = 1
    mask01[5:8, 1, 1:3] = 1
    mask02 = np.where(mask01 == 1, 2, 0).astype(np.int32)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        def write_synthetic(array: np.ndarray, name: str) -> MaskInfo:
            info = MaskInfo(path=tmp_dir / name, array=array,
                            spacing=spacing, origin=origin, direction=direction)
            _save_with_metadata(array, info, tmp_dir, name)
            return info

        def read_back(p: Path) -> Tuple[np.ndarray, Tuple[float, ...], Tuple[float, ...], Tuple[float, ...]]:
            with chdir(p.parent):
                image = sitk.ReadImage(p.name)
                array = sitk.GetArrayFromImage(image)
            return (
                array,
                tuple(float(s) for s in image.GetSpacing()),
                tuple(float(o) for o in image.GetOrigin()),
                tuple(float(d) for d in image.GetDirection()),
            )

        write_synthetic(mask01, "SC_a_ME_label.nii.gz")
        write_synthetic(mask02, "SC_b_ME_label2.nii.gz")

        # --- 0) naming rules (pure strings, legacy-compatible + collision-proof)
        record("naming: case_01_ME_label.nii.gz -> case_01_ME_1_label.nii.gz (legacy-compatible)",
               split_output_name("case_01_ME_label.nii.gz", 1, "_label.nii.gz")
               == "case_01_ME_1_label.nii.gz",
               split_output_name("case_01_ME_label.nii.gz", 1, "_label.nii.gz"))
        record("naming: _label2 keeps the draw token (collision-proof)",
               split_output_name("case_01_ME_label2.nii.gz", 1, "_label.nii.gz")
               == "case_01_ME_label2_1_label.nii.gz",
               split_output_name("case_01_ME_label2.nii.gz", 1, "_label.nii.gz"))
        record("naming: per-vertebra mask never self-overwrites",
               split_output_name("case_01_P_1_label.nii.gz", 3, "_label.nii.gz")
               == "case_01_P_1_3_label.nii.gz",
               split_output_name("case_01_P_1_label.nii.gz", 3, "_label.nii.gz"))
        record("discovery: _label_new/_relabeled excluded, _label2/per-vertebra matched",
               MASK_NAME_PATTERN.match("SC_a_ME_label_new.nii.gz") is None
               and MASK_NAME_PATTERN.match("SC_a_ME_label_relabeled.nii.gz") is None
               and MASK_NAME_PATTERN.match("SC_a_ME_label2.nii.gz") is not None
               and MASK_NAME_PATTERN.match("case_01_P_1_label.nii.gz") is not None,
               "pattern ok")

        # --- 1) split: per-slice (1,y,x), origin z offset, metadata preserved
        split_res = split_mask_along_z(tmp_dir / "SC_a_ME_label.nii.gz")
        record("split: number of outputs = z dimension", len(split_res.output_files) == shape[0],
               f"n={len(split_res.output_files)} vs {shape[0]}")
        record("split: naming matches the legacy rule",
               [p.name for p in split_res.output_files]
               == [f"SC_a_ME_{i + 1}_label.nii.gz" for i in range(shape[0])],
               split_res.output_files[0].name if split_res.output_files else "none")
        ok_split = len(split_res.output_files) == shape[0]
        for i, p in enumerate(split_res.output_files):
            arr, sp, org, dr = read_back(p)
            ok_split &= (
                arr.shape == (1, shape[1], shape[2])
                and np.array_equal(arr, mask01[i:i + 1, :, :])
                and close(sp, spacing)
                and close(org, (*origin[:2], origin[2] + i * spacing[2]))
                and close(dr, direction)
            )
        record("split: per-slice content/spacing/origin(z offset)/direction preserved exactly",
               ok_split, "all slices match" if ok_split else "slice mismatch detected")

        # --- 2) relabel: auto target label, z-continuity runs
        rel_res = relabel_z_continuity(tmp_dir / "SC_a_ME_label.nii.gz")
        record("relabel: output name {stem}_relabeled.nii.gz",
               rel_res.output_path.name == "SC_a_ME_label_relabeled.nii.gz",
               rel_res.output_path.name)
        record("relabel: auto target label = 1", rel_res.target_label == 1,
               f"target={rel_res.target_label}")
        record("relabel: two runs detected z=0~2 and z=5~7",
               [(g.start_z, g.end_z) for g in rel_res.groups] == [(0, 2), (5, 7)],
               str([(g.start_z, g.end_z) for g in rel_res.groups]))
        arr_r, *_ = read_back(rel_res.output_path)
        expected_r = np.zeros(shape, dtype=np.int32)
        expected_r[0:3] = np.where(mask01[0:3] == 1, 1, 0)
        expected_r[5:8] = np.where(mask01[5:8] == 1, 2, 0)
        record("relabel: runs relabeled 1,2 in order; outside = 0",
               np.array_equal(arr_r, expected_r)
               and set(np.unique(arr_r).tolist()) == {0, 1, 2},
               f"unique={np.unique(arr_r).tolist()}")

        # --- 3) relabel: {0,2} mask auto compatibility
        rel2 = relabel_z_continuity(tmp_dir / "SC_b_ME_label2.nii.gz")
        arr_r2, *_ = read_back(rel2.output_path)
        record("relabel: {0,2} mask auto target label = 2",
               rel2.target_label == 2, f"target={rel2.target_label}")
        record("relabel: {0,2} mask yields the same 2 runs and relabeling",
               np.array_equal(arr_r2, expected_r)
               and rel2.output_path.name == "SC_b_ME_label2_relabeled.nii.gz",
               rel2.output_path.name)

        # --- 4) export: full-size binary masks per continuous group
        # Run on SC_b_ME_label2.nii.gz with the default suffix: verifies the
        # default naming (incl. the _label2 collision-proof prefix) and avoids
        # clashing with SC_a split outputs.
        exp_res = export_z_groups(tmp_dir / "SC_b_ME_label2.nii.gz")
        record("export: 2 group files produced", len(exp_res.output_files) == 2,
               f"n={len(exp_res.output_files)}")
        record("export: default-suffix naming SC_b_ME_label2_1_label.nii.gz / _2_",
               [p.name for p in exp_res.output_files]
               == ["SC_b_ME_label2_1_label.nii.gz", "SC_b_ME_label2_2_label.nii.gz"],
               str([p.name for p in exp_res.output_files]))
        ok_exp = len(exp_res.output_files) == 2
        for idx, p in enumerate(exp_res.output_files, start=1):
            arr_e, sp_e, org_e, dr_e = read_back(p)
            g = exp_res.groups[idx - 1]
            expected_e = np.zeros(shape, dtype=np.int32)
            for z in range(g.start_z, g.end_z + 1):
                expected_e[z, :, :] = np.where(mask02[z, :, :] == 2, 1, 0)
            ok_exp &= (
                np.array_equal(arr_e, expected_e)
                and close(sp_e, spacing) and close(org_e, origin) and close(dr_e, direction)
            )
        record("export: group content binary inside run / zero outside; full-size metadata kept",
               ok_exp, "all groups match" if ok_exp else "group mismatch detected")

        # --- 5) overwrite guard
        try:
            split_mask_along_z(tmp_dir / "SC_a_ME_label.nii.gz")
            record("guard: existing split output refused by default", False,
                   "no FileExistsError raised")
        except FileExistsError:
            record("guard: existing split output refused by default", True, "FileExistsError")
        try:
            relabel_z_continuity(tmp_dir / "SC_a_ME_label.nii.gz")
            record("guard: existing relabel output refused by default", False,
                   "no FileExistsError raised")
        except FileExistsError:
            record("guard: existing relabel output refused by default", True, "FileExistsError")
        res_ow = split_mask_along_z(tmp_dir / "SC_a_ME_label.nii.gz", overwrite=True)
        record("guard: overwrite=True allows rewriting",
               len(res_ow.output_files) == shape[0],
               f"n={len(res_ow.output_files)}")

        # --- 6) batch discovery: sources included, relabeled outputs excluded
        found = discover_masks(tmp_dir)
        names = {p.name for p in found}
        record("discovery: discover_masks keeps sources, excludes _relabeled outputs",
               {"SC_a_ME_label.nii.gz", "SC_b_ME_label2.nii.gz"} <= names
               and "SC_a_ME_label_relabeled.nii.gz" not in names,
               f"n_found={len(found)}")

    # --- 7) target-label resolution error paths
    try:
        resolve_target_label(np.zeros(shape, dtype=np.int32))
        record("auto: all-zero mask raises", False, "no ValueError raised")
    except ValueError:
        record("auto: all-zero mask raises", True, "ValueError")
    multi = np.zeros(shape, dtype=np.int32)
    multi[0, 0, 0] = 1
    multi[1, 0, 0] = 2
    try:
        resolve_target_label(multi)
        record("auto: multiple non-zero values require explicit label", False,
               "no ValueError raised")
    except ValueError:
        record("auto: multiple non-zero values require explicit label", True, "ValueError")
    record("explicit: target_label=2 used as-is", resolve_target_label(multi, 2) == 2, "target=2")
    record("CLI parsing: 'auto'/int/invalid",
           parse_target_label_arg("auto") == "auto"
           and parse_target_label_arg("2") == 2,
           "auto/2 ok")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for the zsplit module CLI."""
    parser = argparse.ArgumentParser(
        description="z-axis split / continuity relabel / label-group export for ROI masks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  verify                                        synthetic self-check
  split   --mask case_01_ME_label.nii.gz        per-slice split of one mask
  split   --root ./cases [--recursive]          batch split
  relabel --mask case_01_ME_label.nii.gz        relabel z-continuity runs 1,2,3...
  export  --mask case_01_ME_label.nii.gz        export runs as full-size binary masks
""",
    )
    parser.add_argument("command", choices=["verify", "split", "relabel", "export"],
                        help="verify=self-check; split=per-slice; relabel=continuity runs; export=group binaries")
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--mask", type=Path, help="single mask file path")
    src.add_argument("--root", type=Path, help="batch root directory (auto-discovers *_label*.nii.gz)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="output directory; default: same directory as each mask")
    parser.add_argument("--output-suffix", default=None,
                        help="output filename suffix (split/export default _label.nii.gz; relabel default _relabeled)")
    parser.add_argument("--target-label", default="auto",
                        help="target label: auto=unique non-zero value (default, covers {0,1}/{0,2}); or an integer (1/2)")
    parser.add_argument("--recursive", action="store_true",
                        help="root mode: recurse subdirectories (default: root + direct children)")
    parser.add_argument("--overwrite", action="store_true",
                        help="allow overwriting existing outputs (refused by default)")
    parser.add_argument("--log-dir", type=Path, default=None,
                        help="log file directory; default: console only")
    parser.add_argument("--case-list", type=Path, default=None,
                        help="root mode: text file of case directory names (one per line), for batched runs")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point.

    Args:
        argv: command-line arguments; None = sys.argv.

    Returns:
        Exit code: 0 success; 2 argument/data error; 1 self-check failure or
        batch with failures.
    """
    args = build_parser().parse_args(argv)

    if args.command == "verify":
        checks = self_check()
        n_fail = len([1 for _, ok, _ in checks if not ok])
        for name, ok, msg in checks:
            print(f"{'[PASS]' if ok else '[FAIL]'} {name}: {msg}")
        print(f"summary: {'all checks passed' if n_fail == 0 else f'{n_fail} check(s) FAILED'}")
        return 0 if n_fail == 0 else 1

    logger = setup_logging(args.log_dir)
    logger.info("=" * 60)
    logger.info(f"z-axis mask processing start | command={args.command}")
    logger.info("=" * 60)

    try:
        target_label = parse_target_label_arg(args.target_label)
    except ValueError as e:
        logger.error(str(e))
        return 2

    if not args.mask and not args.root:
        logger.error("split/relabel/export requires --mask or --root")
        return 2

    # Case list (batched runs); ignored in --mask single-file mode
    case_list: Optional[List[str]] = None
    if args.case_list:
        if not args.case_list.is_file():
            logger.error(f"case list not found: {args.case_list}")
            return 2
        case_list = [ln.strip() for ln in
                     args.case_list.read_text(encoding="utf-8").splitlines()
                     if ln.strip() and not ln.strip().startswith("#")]
        logger.info(f"[info] case list: {args.case_list} ({len(case_list)} case(s))")

    if args.command in ("split", "export"):
        suffix = args.output_suffix if args.output_suffix else DEFAULT_SPLIT_SUFFIX
    else:
        suffix = args.output_suffix if args.output_suffix else DEFAULT_RELABEL_SUFFIX

    try:
        if args.mask:
            if args.command == "split":
                res = split_mask_along_z(args.mask, args.output_dir, suffix, args.overwrite, logger)
                n_out = len(res.output_files)
            elif args.command == "relabel":
                res = relabel_z_continuity(args.mask, args.output_dir, suffix,
                                           target_label, args.overwrite, logger)
                n_out = 1
            else:
                res = export_z_groups(args.mask, args.output_dir, suffix,
                                      target_label, args.overwrite, logger)
                n_out = len(res.output_files)
            logger.info(f"[done] {args.mask.name}: produced {n_out} file(s)")
            return 0

        if args.command == "split":
            batch = split_tree(args.root, args.output_dir, suffix, args.recursive,
                               args.overwrite, logger, case_list)
        elif args.command == "relabel":
            batch = relabel_tree(args.root, args.output_dir, suffix, target_label,
                                 args.recursive, args.overwrite, logger, case_list)
        else:
            batch = export_tree(args.root, args.output_dir, suffix, target_label,
                                args.recursive, args.overwrite, logger, case_list)
        logger.info(
            f"[done] {len(batch.ok)} mask(s) ok, {len(batch.failed)} failed, "
            f"{len(batch.outputs)} file(s) produced"
        )
        for p, err in batch.failed:
            logger.error(f"failed: {p}: {err}")
        return 1 if batch.failed else 0
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
