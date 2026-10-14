"""Generate purely SYNTHETIC lumbar-paraspinal-style sample data.

For users who cannot share real imaging: produces small NIfTI phantoms with
the same naming convention as the shipped de-identified samples
(``case_NN_image.nii.gz`` / ``case_NN_ME_label.nii.gz`` /
``case_NN_P_label.nii.gz``) so every toolkit module and ``run_demo.py`` can
be exercised without any real data.

The image is the synthetic two-tissue phantom with a known smooth
multiplicative bias field (see ``lumbarmri.preprocess.synthetic_phantom``);
ME/P labels are two disjoint rectangular ROIs inside the tissue.  The output
is deterministic per ``--seed``.

``--output-dir`` is REQUIRED on purpose: the shipped
``examples/sample_data/`` holds the de-identified real samples and must not
be silently overwritten.

Usage::

    python examples/make_sample_data.py --output-dir ../outputs/synth_samples
    python examples/make_sample_data.py --output-dir /tmp/synth --n-cases 5 --seed 7
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lumbarmri.io import write_image  # noqa: E402
from lumbarmri.preprocess import synthetic_phantom  # noqa: E402


def _two_label_masks(mask_arr: np.ndarray):
    """Split the phantom tissue ROI into two disjoint rectangular labels."""
    z_lim, y_lim, x_lim = mask_arr.shape
    me = np.zeros_like(mask_arr, dtype=np.uint8)
    p = np.zeros_like(mask_arr, dtype=np.uint8)
    y_mid = y_lim // 2
    me[:, :y_mid, :] = mask_arr[:, :y_mid, :]
    p[:, y_mid:, :] = mask_arr[:, y_mid:, :]
    return me, p


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="REQUIRED output directory (e.g. ../outputs/synth_samples); "
             "never point this at examples/sample_data",
    )
    parser.add_argument("--n-cases", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    if args.output_dir.resolve() == (ROOT / "examples" / "sample_data").resolve():
        parser.error("refusing to write into examples/sample_data (real samples live there)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for i in range(args.n_cases):
        case = f"case_{i + 1:02d}"
        img, mask, _true_log = synthetic_phantom(
            two_tissue=True, seed=args.seed + i, shape=(6, 96, 96)
        )
        mask_arr = sitk.GetArrayFromImage(mask)
        me, p = _two_label_masks(mask_arr)

        write_image(img, args.output_dir / f"{case}_image.nii.gz")
        me_img = sitk.GetImageFromArray(me)
        me_img.CopyInformation(img)
        p_img = sitk.GetImageFromArray(p)
        p_img.CopyInformation(img)
        write_image(me_img, args.output_dir / f"{case}_ME_label.nii.gz")
        write_image(p_img, args.output_dir / f"{case}_P_label.nii.gz")
        print(f"wrote {case}_image/ME_label/P_label (.nii.gz)")

    print(f"done: {args.n_cases} synthetic case(s) in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
