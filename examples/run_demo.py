"""Five-module end-to-end demo on the shipped sample data.

Chains every module on ``examples/sample_data/case_01`` + the synthetic demo
clinic table:

1. N4 bias-field correction (per-slice 2D, ROI estimation mask)
2. ROI z-score normalization
3. Two-component GMM fat/muscle segmentation (FM_ratio) for ME and P
4. z-axis mask splitting of the ME label
5. BIC-LASSO variable selection (coefficient path + BIC curve)

All products land in ``outputs/demo_run/`` (git-ignored).  Every step
asserts its outputs exist; the script exits 0 on full success.

Usage::

    python examples/run_demo.py
    python examples/run_demo.py --quick     # skip the LASSO step
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lumbarmri.bic_lasso import load_config, run as lasso_run  # noqa: E402
from lumbarmri.gmm_segmentation import RoiResult, run_gmm_on_roi, write_stats_csv  # noqa: E402
from lumbarmri.io import read_image, write_image  # noqa: E402
from lumbarmri.normalization import zscore  # noqa: E402
from lumbarmri.preprocess import N4Params, correct  # noqa: E402
from lumbarmri.zsplit import split_mask_along_z  # noqa: E402

SAMPLE = ROOT / "examples" / "sample_data"
OUT = ROOT / "outputs" / "demo_run"


def _ok(step: str, msg: str) -> None:
    print(f"  [OK] {step}: {msg}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="skip the LASSO step")
    args = parser.parse_args(argv)

    t0 = time.time()
    print(f"== lumbar-gmm-fmr-toolkit demo ==\noutputs -> {OUT}\n")

    image_path = SAMPLE / "case_01_image.nii.gz"
    me_path = SAMPLE / "case_01_ME_label.nii.gz"
    p_path = SAMPLE / "case_01_P_label.nii.gz"
    for p in (image_path, me_path, p_path):
        if not p.is_file():
            print(f"missing sample file: {p} (see examples/README.md)")
            return 2

    # ---- 1. N4 bias-field correction (per-slice, ROI mask) -------------
    print("[1/5] N4 bias-field correction (2D per-slice)")
    img = read_image(image_path)
    me = read_image(me_path)
    res = correct(img, roi=me, mode="2d", params=N4Params(), mask_method="roi")
    n4_out = OUT / "n4" / "case_01_image_n4.nii.gz"
    write_image(res.corrected, n4_out)
    assert n4_out.is_file()
    _ok("N4", f"CV {res.cv_before:.5f} -> {res.cv_after:.5f} | wrote {n4_out.name}")

    # ---- 2. ROI z-score normalization ----------------------------------
    print("[2/5] ROI z-score normalization")
    zimg = zscore(res.corrected, me)
    z_out = OUT / "norm" / "case_01_image_n4_z.nii.gz"
    write_image(zimg, z_out)
    z_arr = sitk.GetArrayViewFromImage(zimg)
    me_arr = sitk.GetArrayViewFromImage(me)
    assert z_out.is_file()
    assert abs(float(z_arr[me_arr > 0].mean())) < 1e-3
    _ok("z-score", f"in-ROI mean ~0 | wrote {z_out.name}")

    # ---- 3. GMM fat/muscle segmentation (ME and P) ----------------------
    print("[3/5] Two-component GMM segmentation")
    corrected_arr = sitk.GetArrayFromImage(res.corrected)
    spacing = res.corrected.GetSpacing()
    # SimpleITK spacing is (x, y, z); arrays are (z, y, x)
    spacing_zyx = (spacing[2], spacing[1], spacing[0])
    for name, mask_path in (("ME", me_path), ("P", p_path)):
        mask_arr = sitk.GetArrayFromImage(read_image(mask_path))
        clusters, n_vox, degenerate = run_gmm_on_roi(
            corrected_arr, mask_arr, spacing_zyx
        )
        result = RoiResult(
            image_path=n4_out,
            mask_path=mask_path,
            output_mask=OUT / "gmm" / f"case_01_{name}_gmm.nii.gz",
            stats_csv=OUT / "gmm" / f"case_01_{name}_stats.csv",
            clusters=clusters,
            roi_voxels=n_vox,
            degenerate=degenerate,
        )
        csv_path = write_stats_csv(result, result.stats_csv)
        assert csv_path.is_file()
        _ok(f"GMM {name}", f"FM_ratio={result.fm_ratio:.4f} over {n_vox} voxels | {csv_path.name}")

    # ---- 4. z-axis mask splitting ---------------------------------------
    print("[4/5] z-axis mask splitting")
    split = split_mask_along_z(
        me_path, output_dir=OUT / "zsplit", overwrite=True
    )
    assert split.output_files and all(f.is_file() for f in split.output_files)
    _ok("zsplit", f"{len(split.output_files)} slice file(s) in {split.output_files[0].parent.name}/")

    # ---- 5. BIC-LASSO variable selection ---------------------------------
    if args.quick:
        print("[5/5] BIC-LASSO skipped (--quick)")
    else:
        print("[5/5] BIC-LASSO variable selection (synthetic demo table)")
        cfg = load_config(ROOT / "configs" / "lasso_demo.yaml")
        cfg.output_dir = OUT / "lasso"
        summary = lasso_run(cfg, n_bootstrap=30)
        fig3 = cfg.output_dir / "fig3_lasso_coef_path.png"
        fig4 = cfg.output_dir / "fig4_bic_criterion.png"
        assert fig3.is_file() and fig4.is_file()
        _ok("BIC-LASSO", f"lambda={summary['lambda_grid_argmin']:.6f} | figures written")

    print(f"\n== demo finished in {time.time() - t0:.1f} s, all steps OK ==")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
