"""Two-component GMM segmentation of lumbar paraspinal muscle ROIs (fat/muscle).

Background
----------
Run an unsupervised Gaussian Mixture Model (GMM, K=2) on the voxel
intensities inside an ROI mask (``*_label.nii.gz``), splitting the ROI voxels
into fat and muscle clusters, then report per-cluster volume/intensity
statistics and the study's primary imaging metric:

    FM_ratio = V_fat / (V_fat + V_muscle)

The fat cluster is the cluster with the **largest** ``mean_gray`` (raw
intensity mean; fat is hyperintense on T2-weighted images).  GMM parameters
are frozen to match the original research pipeline exactly:

    ROI voxel intensities (mask > 0) -> StandardScaler
    -> GaussianMixture(n_components=2, random_state=42)   # covariance_type 'full'
    -> predict + 1 -> per-cluster count / volume_mm3 / mean_gray
    -> FM_ratio = volume_mm3 of the max-mean_gray cluster / sum of both volumes

Data compatibility
------------------
- Mask discovery: filename pattern ``{stem}_label{draw}.nii.gz`` is matched
  automatically; no case IDs are hard-coded, so new case directories work
  out of the box.
- Flat layout: ``{id}_image.nii.gz`` + ``{id}_{region}_label.nii.gz``
  (whole segment) and per-vertebra masks ``{id}_{region}_{n}_label.nii.gz``.
- Repeat-draw masks: ``{id}_{region}_label2.nii.gz`` (values {0,2}).
  ROI is always ``mask > 0`` (covers both {0,1} and {0,2} encodings; on
  {0,1} masks this equals the original ``mask == 1`` pipeline).
- Image lookup: strip ``_label{draw}`` from the mask stem, then pick the
  longest-prefix ``*_image.nii.gz`` in the same directory
  (``case_01_P_1_label`` -> ``case_01_image``).
- Historical outputs (``*_label_new.nii.gz``, ``*_relabeled.nii.gz``) do not
  match the discovery pattern, so batch re-runs skip them naturally.
- Degenerate handling: if the ROI has < 2 voxels or a single unique
  intensity, GMM cannot split two classes; the ROI is treated as one cluster
  (``degenerate=True``) and FM_ratio is recorded as 1.0.

Usage
-----
    lumbarmri gmm verify                                  # synthetic self-check
    lumbarmri gmm single --image x_image.nii.gz --mask x_L_label.nii.gz
    lumbarmri gmm case --case-dir case_01                 # one case (auto-discovers labels)
    lumbarmri gmm tree --root ./cases                     # batch of case directories

Safety boundary: source image/mask files are read-only; outputs are only
``{mask_stem}_{suffix}.nii.gz`` and ``{mask_stem}_{suffix}_stats.csv``
(the suffix guarantees they cannot collide with source files).

Import example::

    from lumbarmri.gmm_segmentation import segment_mask
    result = segment_mask(img_path=Path("case_01_image.nii.gz"),
                          mask_path=Path("case_01_ME_label.nii.gz"))
    print(result.fm_ratio, result.clusters)
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import SimpleITK as sitk

from .io import chdir

# ---------------------------------------------------------------------------
# Constants (frozen research conventions -- do not change)
# ---------------------------------------------------------------------------

#: GMM parameters: identical to the original research pipeline
GMM_N_COMPONENTS = 2
GMM_RANDOM_STATE = 42

#: Mask discovery pattern inside a case directory: {any stem}_label{draw}.nii.gz
#: Compatible with whole-segment masks (case_01_ME_label.nii.gz), per-vertebra
#: masks (case_01_P_1_label.nii.gz) and repeat draws (case_01_ME_label2.nii.gz);
#: historical outputs *_label_new.nii.gz / *_relabeled.nii.gz are excluded
#: because they do not match the pattern.
MASK_NAME_PATTERN = re.compile(r"^(?P<stem>.+)_label(?P<draw>\d*)\.nii\.gz$")

#: Suffix used when auto-discovering the image file (same directory; the image
#: stem is a prefix of the mask stem)
DEFAULT_IMG_SUFFIX = "_image.nii.gz"

#: Synthetic self-check volume parameters (fat = high intensity, muscle = low
#: intensity, T2WI direction)
SELF_CHECK_SHAPE = (8, 16, 16)
SELF_CHECK_SPACING = (1.5, 0.7, 0.7)
SELF_CHECK_MUSCLE_LO, SELF_CHECK_MUSCLE_HI = 200.0, 400.0
SELF_CHECK_FAT_LO, SELF_CHECK_FAT_HI = 700.0, 900.0

#: Floating-point comparison tolerances (tiered, shared across the toolkit)
TOL_EXACT = 1e-9
TOL_FLOAT = 1e-3


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ClusterStat:
    """Statistics of a single GMM cluster."""

    label: int  # cluster label (predict + 1)
    count: int  # voxel count
    volume_mm3: float  # count x product of mask spacing
    mean_gray: float  # raw intensity mean (fat cluster = the largest)


@dataclass
class RoiResult:
    """Complete GMM segmentation result for one ROI."""

    image_path: Path
    mask_path: Path
    output_mask: Path
    stats_csv: Path
    clusters: List[ClusterStat] = field(default_factory=list)
    roi_voxels: int = 0
    degenerate: bool = False  # True = single-cluster fallback, FM_ratio = 1.0

    @property
    def fm_ratio(self) -> float:
        """FM_ratio = volume of max-mean_gray cluster / sum of cluster volumes.

        Degenerate results report 1.0.
        """
        if len(self.clusters) < GMM_N_COMPONENTS:
            return 1.0
        fat = max(self.clusters, key=lambda c: c.mean_gray)
        vol_sum = sum(c.volume_mm3 for c in self.clusters)
        return fat.volume_mm3 / vol_sum if vol_sum > 0 else 1.0


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
    logger = logging.getLogger("gmm_segmentation")
    logger.setLevel(logging.DEBUG)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"gmm_segmentation_{datetime.now():%Y%m%d_%H%M%S}.log"
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
# Case discovery (no hard-coded IDs; filename-driven, new cases just work)
# ---------------------------------------------------------------------------


def discover_case_masks(case_dir: Path) -> List[Path]:
    """Find all GMM target masks in a case directory.

    Matches the filename pattern ``{stem}_label{draw}.nii.gz`` (``draw`` is
    the annotation round: empty = first, '2' = repeat).  No region/ID
    filtering is hard-coded: whole-segment masks, per-vertebra masks and
    repeat-draw masks are all picked up; historical outputs
    (``*_label_new.nii.gz``, ``*_relabeled.nii.gz``) are excluded because
    they do not match the pattern.

    Args:
        case_dir: case directory (or flat directory).

    Returns:
        Sorted list of mask paths; empty list when nothing matches.

    Raises:
        FileNotFoundError: case_dir is not a directory.
    """
    if not case_dir.is_dir():
        raise FileNotFoundError(f"case directory not found: {case_dir}")

    return sorted(
        entry
        for entry in case_dir.iterdir()
        if entry.is_file() and MASK_NAME_PATTERN.match(entry.name)
    )


def find_image_for(mask_path: Path) -> Path:
    """Locate the image file for a mask (same directory, stem-prefix match).

    Strips ``_label{draw}`` from the mask name to get the mask stem; among all
    ``*_image.nii.gz`` files in the same directory, selects the one whose stem
    is a prefix of the mask stem (longest wins).  Compatible with:

    - ``case_01_ME_label.nii.gz`` -> ``case_01_image.nii.gz``;
    - ``case_01_P_1_label.nii.gz`` (per-vertebra) -> ``case_01_image.nii.gz``.

    Args:
        mask_path: mask file path.

    Returns:
        Image file path.

    Raises:
        FileNotFoundError: mask name does not match the pattern, or no image
            file was found for it.
    """
    m = MASK_NAME_PATTERN.match(mask_path.name)
    if not m:
        raise FileNotFoundError(
            f"mask name does not match the *_label.nii.gz pattern: {mask_path.name}"
        )
    mask_stem = m.group("stem")

    candidates = [
        entry
        for entry in mask_path.parent.glob("*_image.nii.gz")
        if mask_stem.startswith(entry.name[: -len(DEFAULT_IMG_SUFFIX)] + "_")
    ]
    if not candidates:
        raise FileNotFoundError(
            f"no image file found for mask stem '{mask_stem}' "
            f"(no prefix-matching *_image.nii.gz in {mask_path.parent}); "
            f"derived from {mask_path.name}"
        )
    return max(candidates, key=lambda p: len(p.name))


def iter_case_dirs(root_dir: Path, recursive: bool = False,
                   case_list: Optional[Sequence[str]] = None) -> List[Path]:
    """Enumerate case directories to process under root_dir.

    - ``recursive=False``: flat masks in root_dir or its direct subdirectories;
    - ``recursive=True``: every directory at any depth that contains masks
      (deduplicated);
    - ``case_list``: keep only these directory names (for batched runs).

    Args:
        root_dir: batch root directory.
        recursive: whether to recurse into subdirectories.
        case_list: directory-name whitelist; None = no filtering.

    Returns:
        List of directories (root_dir itself is included when it holds flat
        masks).

    Raises:
        FileNotFoundError: root_dir is not a directory.
    """
    if not root_dir.is_dir():
        raise FileNotFoundError(f"root directory not found: {root_dir}")

    wanted = set(case_list) if case_list is not None else None

    def keep(d: Path) -> bool:
        return wanted is None or d.name in wanted or d.name.removeprefix("STU_") in wanted

    dirs: List[Path] = []
    if keep(root_dir) and discover_case_masks(root_dir):
        dirs.append(root_dir)
    if recursive:
        for p in sorted(root_dir.rglob("*.nii.gz")):
            if MASK_NAME_PATTERN.match(p.name) and p.parent not in dirs and keep(p.parent):
                dirs.append(p.parent)
    else:
        for child in sorted(root_dir.iterdir()):
            if child.is_dir() and keep(child) and discover_case_masks(child) and child not in dirs:
                dirs.append(child)
    return dirs


# ---------------------------------------------------------------------------
# GMM core: one fit feeds both the written label map and the cluster stats
# (parameter conventions identical to the original pipeline)
# ---------------------------------------------------------------------------


def _fit_gmm(gray_values: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Fit GMM once on an ROI intensity sequence.

    Degenerate when voxel count < 2 or the intensity is constant (GMM cannot
    separate two classes).

    Args:
        gray_values: 1-D array of ROI voxel intensities.

    Returns:
        (labels after predict + 1, degenerate flag).  Labels are empty when
        degenerate.
    """
    if gray_values.size < GMM_N_COMPONENTS or np.std(gray_values) == 0:
        return np.array([], dtype=int), True
    scaler = StandardScaler()
    scaled = scaler.fit_transform(gray_values.reshape(-1, 1))
    gmm = GaussianMixture(n_components=GMM_N_COMPONENTS, random_state=GMM_RANDOM_STATE)
    gmm.fit(scaled)
    return gmm.predict(scaled) + 1, False


def _cluster_stats(labels: np.ndarray, gray_values: np.ndarray,
                   voxel_volume: float, degenerate: bool) -> List[ClusterStat]:
    """Per-cluster count / volume_mm3 / mean_gray from labels and intensities.

    Args:
        labels: labels after predict + 1; ignored when degenerate.
        gray_values: 1-D ROI intensities (same order as labels).
        voxel_volume: single-voxel volume (mm^3).
        degenerate: single-cluster fallback flag.

    Returns:
        ClusterStat list sorted by label ascending.
    """
    if degenerate:
        return [ClusterStat(1, int(gray_values.size), int(gray_values.size) * voxel_volume,
                            float(gray_values.mean()) if gray_values.size else float("nan"))]
    df = pd.DataFrame({"gray": gray_values, "labels": labels})
    counts = df["labels"].value_counts().sort_index()
    means = df.groupby("labels")["gray"].mean().sort_index()
    return [
        ClusterStat(int(label), int(count), int(count) * voxel_volume, float(mean))
        for label, count, mean in zip(counts.index, counts.values, means.values)
    ]


def run_gmm_on_roi(
    image: np.ndarray, mask: np.ndarray, spacing: Tuple[float, float, float]
) -> Tuple[List[ClusterStat], int, bool]:
    """Run GMM clustering statistics on one ROI (in memory, no disk I/O).

    Pipeline: ROI intensities -> StandardScaler -> GaussianMixture(2,
    random_state=42) -> predict + 1 -> per-cluster count / volume_mm3 /
    mean_gray.  One convention difference from the historical pipeline: the
    ROI is ``mask > 0`` rather than ``mask == 1`` (so repeat-draw masks with
    value 2 are covered; on {0,1} masks the two are equivalent).

    Args:
        image: image array (z, y, x).
        mask: ROI mask array; values {0,1} (first draw) or {0,2} (repeat).
        spacing: mask spacing in three axes, used for volume_mm3 = count x product.

    Returns:
        (cluster stats, ROI voxel count, degenerate flag).

    Raises:
        ValueError: image/mask shapes differ, or mask values are illegal.
    """
    if image.shape != mask.shape:
        raise ValueError(f"image/mask shape mismatch: {image.shape} vs {mask.shape}")
    uniq = np.unique(mask)
    if not set(uniq.tolist()) <= {0, 1} and not set(uniq.tolist()) <= {0, 2}:
        raise ValueError(f"illegal mask values (expected {{0,1}} or {{0,2}}): {uniq.tolist()}")

    gray_values = image[mask > 0]
    voxel_volume = float(spacing[0]) * float(spacing[1]) * float(spacing[2])
    labels, degenerate = _fit_gmm(gray_values)
    clusters = _cluster_stats(labels, gray_values, voxel_volume, degenerate)
    return clusters, int(gray_values.size), degenerate


# ---------------------------------------------------------------------------
# Segmentation and statistics output
# ---------------------------------------------------------------------------


def segment_mask(
    img_path: Path,
    mask_path: Path,
    output_dir: Optional[Path] = None,
    suffix: str = "new",
    logger: Optional[logging.Logger] = None,
) -> RoiResult:
    """Segment one image/mask pair with GMM and write mask + stats CSV.

    The labels from a single GMM fit are used both for the written label map
    and for the cluster statistics, so the two stay strictly consistent.

    Args:
        img_path: image file path (``.nii.gz``).
        mask_path: mask file path (``.nii.gz``).
        output_dir: output directory; None = the mask's own directory.
        suffix: output filename suffix (default 'new', never collides with
            source files).
        logger: optional logger.

    Returns:
        RoiResult with output paths, cluster stats and FM_ratio.

    Raises:
        FileNotFoundError: image or mask missing.
        ValueError: image/mask shapes differ.
    """
    img_path, mask_path = Path(img_path), Path(mask_path)
    if not img_path.is_file():
        raise FileNotFoundError(f"image file not found: {img_path}")
    if not mask_path.is_file():
        raise FileNotFoundError(f"mask file not found: {mask_path}")

    out_dir = Path(output_dir) if output_dir else mask_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = mask_path.name.replace(".nii.gz", "")
    output_mask = out_dir / f"{stem}_{suffix}.nii.gz"
    stats_csv = out_dir / f"{stem}_{suffix}_stats.csv"

    # chdir channel for non-ASCII-safe SimpleITK I/O
    with chdir(mask_path.parent):
        mask_image = sitk.ReadImage(mask_path.name)
        mask_array = sitk.GetArrayFromImage(mask_image)
        spacing = tuple(float(s) for s in mask_image.GetSpacing())
    with chdir(img_path.parent):
        image_array = sitk.GetArrayFromImage(sitk.ReadImage(img_path.name))

    if image_array.shape != mask_array.shape:
        raise ValueError(
            f"image/mask shape mismatch: {image_array.shape} vs {mask_array.shape}"
        )

    roi = mask_array > 0
    gray_values = image_array[roi]
    voxel_volume = spacing[0] * spacing[1] * spacing[2]
    labels, degenerate = _fit_gmm(gray_values)
    clusters = _cluster_stats(labels, gray_values, voxel_volume, degenerate)

    # Write new label map: ROI voxels get cluster labels (1 when degenerate);
    # spatial metadata preserved.
    new_array = np.zeros_like(mask_array, dtype=np.int32)
    new_array[roi] = 1 if degenerate else labels
    new_seg = sitk.GetImageFromArray(new_array)
    new_seg.SetOrigin(mask_image.GetOrigin())
    new_seg.SetSpacing(mask_image.GetSpacing())
    new_seg.SetDirection(mask_image.GetDirection())
    with chdir(out_dir):
        sitk.WriteImage(new_seg, output_mask.name)

    result = RoiResult(
        image_path=img_path,
        mask_path=mask_path,
        output_mask=output_mask,
        stats_csv=stats_csv,
        clusters=clusters,
        roi_voxels=int(gray_values.size),
        degenerate=degenerate,
    )
    write_stats_csv(result, stats_csv)

    if logger:
        logger.info(
            f"{mask_path.name}: {len(clusters)} clusters, roi_voxels={result.roi_voxels}, "
            f"degenerate={degenerate}, FM_ratio={result.fm_ratio:.6f} -> {output_mask.name}"
        )
    return result


def write_stats_csv(result: RoiResult, out_path: Path) -> Path:
    """Write cluster stats CSV (label/count/volume_mm3/mean_gray/FM_ratio).

    Args:
        result: result of segment_mask.
        out_path: CSV output path.

    Returns:
        out_path.
    """
    df = pd.DataFrame(
        [
            {
                "image_path": str(result.image_path),
                "mask_path": str(result.mask_path),
                "label": c.label,
                "count": c.count,
                "volume_mm3": f"{c.volume_mm3:.6f}",
                "mean_gray": f"{c.mean_gray:.4f}",
                "FM_ratio": f"{result.fm_ratio:.6f}",
                "degenerate": int(result.degenerate),
            }
            for c in result.clusters
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with chdir(out_path.parent):
        df.to_csv(out_path.name, index=False, encoding="utf-8-sig")
    return out_path


def segment_case(
    case_dir: Path,
    output_dir: Optional[Path] = None,
    suffix: str = "new",
    logger: Optional[logging.Logger] = None,
    exclude_substr: Sequence[str] = (),
) -> List[RoiResult]:
    """Process one case directory (auto-discovers all _label/_label2 masks).

    Args:
        case_dir: case directory; a flat image+label directory also works.
        output_dir: output directory; None = each mask's own directory.
        suffix: output filename suffix.
        logger: optional logger.
        exclude_substr: skip masks whose filename contains any of these
            substrings.

    Returns:
        RoiResult list in discovery order.

    Raises:
        FileNotFoundError: directory missing, or a mask has no image partner.
    """
    case_dir = Path(case_dir)
    results: List[RoiResult] = []
    n_skipped = 0
    for mask_path in discover_case_masks(case_dir):
        if any(s and s in mask_path.name for s in exclude_substr):
            n_skipped += 1
            if logger:
                logger.info(f"skip {mask_path.name} (matched exclude_substr)")
            continue
        img_path = find_image_for(mask_path)
        results.append(segment_mask(img_path, mask_path, output_dir, suffix, logger))
    if n_skipped and logger:
        logger.info(f"[skip] {case_dir.name}: excluded {n_skipped} mask(s) via exclude_substr")
    return results


def segment_tree(
    root_dir: Path,
    output_dir: Optional[Path] = None,
    suffix: str = "new",
    recursive: bool = False,
    logger: Optional[logging.Logger] = None,
    exclude_substr: Sequence[str] = (),
    case_list: Optional[Sequence[str]] = None,
) -> List[RoiResult]:
    """Batch-process every case directory under root_dir.

    Args:
        root_dir: batch root directory.
        output_dir: output directory; None = each mask's own directory.
        suffix: output filename suffix.
        recursive: recurse into all levels of mask-containing directories.
        logger: optional logger.
        exclude_substr: forwarded to segment_case.
        case_list: process only these directory names (batched runs).

    Returns:
        All RoiResults.

    Raises:
        FileNotFoundError: root directory missing.
    """
    results: List[RoiResult] = []
    for case_dir in iter_case_dirs(root_dir, recursive, case_list):
        results.extend(segment_case(case_dir, output_dir, suffix, logger, exclude_substr))
    return results


# ---------------------------------------------------------------------------
# Self-check (synthetic data; no external dataset needed)
# ---------------------------------------------------------------------------


def self_check() -> List[Tuple[str, bool, str]]:
    """GMM implementation self-check on synthetic images.

    Verifies the frozen parameter conventions, the degenerate branch and
    {0,2} masks.

    Returns:
        [(check name, passed, detail)] list.
    """
    results: List[Tuple[str, bool, str]] = []

    def record(name: str, ok: bool, msg: str) -> None:
        results.append((name, bool(ok), msg))

    rng = np.random.default_rng(20261002)
    shape = SELF_CHECK_SHAPE
    image = np.zeros(shape, dtype=float)
    mask = np.zeros(shape, dtype=np.int32)
    # Two ROIs: x < half-width gets value 1 (first draw), x == half-width gets
    # value 2 (repeat draw).  Each ROI voxel is fat (high-intensity cluster)
    # with probability 1/3, muscle (low-intensity cluster) otherwise.
    n_fat = 0
    n_mus = 0
    fat_sel = rng.random(shape) < 1.0 / 3.0
    for z in range(shape[0]):
        for y in range(shape[1]):
            for x in range(shape[2]):
                if x < shape[2] // 2:
                    mask[z, y, x] = 1
                elif x == shape[2] // 2:
                    mask[z, y, x] = 2
                else:
                    continue
                if fat_sel[z, y, x]:
                    image[z, y, x] = rng.uniform(SELF_CHECK_FAT_LO, SELF_CHECK_FAT_HI)
                    n_fat += 1
                else:
                    image[z, y, x] = rng.uniform(SELF_CHECK_MUSCLE_LO, SELF_CHECK_MUSCLE_HI)
                    n_mus += 1
    true_fm = n_fat / (n_fat + n_mus)
    spacing = SELF_CHECK_SPACING
    voxel_vol = spacing[0] * spacing[1] * spacing[2]

    def summarize(clusters: List[ClusterStat], roi_v: int, degen: bool) -> RoiResult:
        return RoiResult(Path("img"), Path("m"), Path("o"), Path("s"),
                         clusters, roi_v, degen)

    # --- 1) {0,1} mask: K=2, volume convention, full ROI, FM_ratio recovers truth
    mask01 = (mask == 1).astype(np.int32)
    clusters, roi_v, degen = run_gmm_on_roi(image, mask01, spacing)
    res01 = summarize(clusters, roi_v, degen)
    record("check1: {0,1} mask yields 2 clusters", len(clusters) == GMM_N_COMPONENTS,
           f"n_clusters={len(clusters)}")
    record("check1: volume_mm3 = count x spacing product",
           all(abs(c.volume_mm3 - c.count * voxel_vol) < TOL_EXACT for c in clusters),
           f"voxel_vol={voxel_vol}")
    record("check1: ROI = mask>0 taken in full", roi_v == int(mask01.sum()),
           f"roi_voxels={roi_v} vs {int(mask01.sum())}")
    record("check1: FM_ratio ~= synthetic fat share (1e-2)",
           abs(res01.fm_ratio - true_fm) < 1e-2,
           f"FM={res01.fm_ratio:.4f} vs true={true_fm:.4f}")
    fat = max(clusters, key=lambda c: c.mean_gray)
    record("check1: fat cluster = max-mean_gray cluster (T2WI high signal)",
           fat.mean_gray > min(c.mean_gray for c in clusters) + 100,
           f"max={fat.mean_gray:.1f} vs min={min(c.mean_gray for c in clusters):.1f}")

    # --- 2) {0,2} mask (repeat draw): full ROI, two clusters as well
    mask02 = np.where(mask == 2, 2, 0).astype(np.int32)
    clusters2, roi_v2, degen2 = run_gmm_on_roi(image, mask02, spacing)
    record("check2: {0,2} mask ROI taken in full", roi_v2 == int((mask02 > 0).sum()),
           f"roi_voxels={roi_v2} vs {int((mask02 > 0).sum())}")
    record("check2: {0,2} mask yields 2 clusters", len(clusters2) == GMM_N_COMPONENTS,
           f"n_clusters={len(clusters2)}")

    # --- 3) Degenerate: constant-intensity ROI -> single cluster, FM_ratio=1.0
    flat_image = np.full(shape, 500.0)
    clusters3, _, degen3 = run_gmm_on_roi(flat_image, mask01, spacing)
    res3 = summarize(clusters3, int(mask01.sum()), degen3)
    record("check3: constant ROI takes degenerate branch", degen3 and len(clusters3) == 1,
           f"degenerate={degen3}, n_clusters={len(clusters3)}")
    record("check3: FM_ratio = 1.0 when degenerate", abs(res3.fm_ratio - 1.0) < TOL_EXACT,
           f"FM={res3.fm_ratio}")

    # --- 4) Directional semantics: high intensity = fat; FM should track the fat
    #        voxel share, not the muscle share
    fat_share = true_fm
    mus_share = 1.0 - true_fm
    record("check4: FM tracks fat voxel share (not muscle share)",
           abs(res01.fm_ratio - fat_share) < abs(res01.fm_ratio - mus_share),
           f"FM={res01.fm_ratio:.4f}, fat={fat_share:.4f}, muscle={mus_share:.4f}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for the gmm module CLI."""
    parser = argparse.ArgumentParser(
        description="Two-component GMM segmentation of paraspinal muscle ROIs (fat/muscle)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  verify                                  synthetic self-check
  single --image x_image.nii.gz --mask x_ME_label.nii.gz
  case   --case-dir case_01
  tree   --root ./cases [--recursive]
""",
    )
    parser.add_argument("command", choices=["verify", "single", "case", "tree"],
                        help="verify=self-check; single=one file pair; case=one case dir; tree=batch")
    parser.add_argument("--image", type=Path, help="single: image file path")
    parser.add_argument("--mask", type=Path, help="single: mask file path")
    parser.add_argument("--case-dir", type=Path, help="case: case directory (auto-discovers labels)")
    parser.add_argument("--root", type=Path, help="tree: batch root directory")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="output directory; default: each mask's own directory")
    parser.add_argument("--suffix", default="new", help="output filename suffix (default: %(default)s)")
    parser.add_argument("--recursive", action="store_true",
                        help="tree: recurse into all mask-containing directories (default: direct children)")
    parser.add_argument("--log-dir", type=Path, default=None,
                        help="log file directory; default: console only")
    parser.add_argument("--exclude-substr", action="append", default=[],
                        help="case/tree: skip masks whose filename contains this substring (repeatable)")
    parser.add_argument("--case-list", type=Path, default=None,
                        help="case/tree: text file of case directory names (one per line), for batched runs")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point.

    Args:
        argv: command-line arguments; None = sys.argv.

    Returns:
        Exit code: 0 success; 2 argument/data error; 1 self-check failure.
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
    logger.info(f"GMM segmentation start | command={args.command} | suffix={args.suffix}")
    logger.info("=" * 60)

    if args.command == "single":
        if not args.image or not args.mask:
            logger.error("'single' requires --image and --mask")
            return 2
        result = segment_mask(args.image, args.mask, args.output_dir, args.suffix, logger)
        logger.info(f"[done] FM_ratio={result.fm_ratio:.6f} -> {result.output_mask}")
        return 0

    if args.command == "case":
        if not args.case_dir:
            logger.error("'case' requires --case-dir")
            return 2
        try:
            results = segment_case(args.case_dir, args.output_dir, args.suffix, logger,
                                   exclude_substr=args.exclude_substr)
        except FileNotFoundError as e:
            logger.error(str(e))
            return 2
        logger.info(f"[done] processed {len(results)} mask(s)")
        return 0

    if args.command == "tree":
        if not args.root:
            logger.error("'tree' requires --root")
            return 2
        try:
            case_list = None
            if args.case_list:
                if not args.case_list.is_file():
                    logger.error(f"case list not found: {args.case_list}")
                    return 2
                case_list = [ln.strip() for ln in
                             args.case_list.read_text(encoding="utf-8").splitlines()
                             if ln.strip() and not ln.strip().startswith("#")]
                logger.info(f"[info] case list: {args.case_list} ({len(case_list)} case(s))")
            results = segment_tree(args.root, args.output_dir, args.suffix,
                                   recursive=args.recursive, logger=logger,
                                   exclude_substr=args.exclude_substr,
                                   case_list=case_list)
        except FileNotFoundError as e:
            logger.error(str(e))
            return 2
        logger.info(f"[done] processed {len(results)} mask(s)")
        return 0

    logger.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
