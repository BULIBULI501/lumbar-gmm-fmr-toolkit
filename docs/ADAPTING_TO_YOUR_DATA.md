# Adapting the toolkit to your own data

Five steps from raw lumbar paraspinal MRIs + a clinic table to published
figures. Each step is one module, one CLI entry point, and is independently
testable (`… verify` runs a synthetic self-check anywhere).

```text
raw NIfTI + ROI masks ──► [1] preprocess (N4) ──► [2] normalization (z-score)
        ──► [3] gmm_segmentation (FM_ratio per ROI) ──► [4] zsplit (per-level)
clinic table (CSV) ─────────────────────────────────► [5] bic_lasso (figures)
```

## Naming convention (read this first)

| piece | rule |
|---|---|
| case file | `<case>_image.nii.gz` |
| ROI mask | `<case>_<region>_label.nii.gz` — suffix is **always `_label`** |
| regions used in the shipped demo | `ME` (multifidus–erector spinae complex), `P` (psoas major) |
| anonymised datasets | `case_NN_image.nii.gz`, `case_NN_ME_label.nii.gz`, `case_NN_P_label.nii.gz` |

Any `*` stem works as long as masks end in `_label.nii.gz`; batch commands
auto-discover `*_label*.nii.gz`. If you receive masks suffixed `_roi`, rename
them first (`python -m lumbarmri.deidentify` does this automatically during
anonymisation).

## Step 1 — N4 bias-field correction (`preprocess`)

```bash
python -m lumbarmri.preprocess single \
    --image case_01_image.nii.gz --mask case_01_ME_label.nii.gz \
    --output-dir outputs/n4 --mode 2d --mask-method roi
```

- Use `--mode 2d` (default) for anisotropic T2WI (thick slices, e.g.
  10–15 mm vs ~0.4 mm in-plane); `3d` for isotropic volumes.
- `--mask-method roi` keeps abdominal fat from dragging the B-spline field;
  see `docs/ALGORITHM_NOTES.md` §1 for the measured comparison.
- Batch: `tree` / `tree-single` over a data root + optional `--case-list`;
  `qc` writes a per-case CV table.

## Step 2 — Intensity normalization (`normalization`)

```bash
python -m lumbarmri.normalization zscore \
    --image outputs/n4/case_01_image.nii.gz \
    --mask case_01_ME_label.nii.gz --output outputs/norm/case_01_image.nii.gz
```

ROI z-score is the default analysis route. Nyul standardization
(`nyul-train` / `nyul-apply`) and histogram matching are available when you
pool multiple scanners. Watch two documented traps: int16 inputs need the
module's explicit float32 output type (already handled), and a
zero-variance ROI keeps the raw gray values instead of returning zeros.

## Step 3 — GMM fat/muscle split (`gmm_segmentation`)

```bash
python -m lumbarmri.gmm_segmentation case --case-dir case_01 --output-dir outputs/gmm
```

For each ROI mask: voxels `mask > 0` → `StandardScaler` →
`GaussianMixture(2, random_state=42, covariance_type='full')` → the cluster
with the higher mean gray is fat →
`FM_ratio = V_fat / (V_fat + V_muscle)`. A degenerate ROI yields
`FM_ratio = 1.0` by definition. The full per-case output includes fat/muscle
volumes in mm³ and the `FM_ratio` column used downstream.

## Step 4 — Z-axis level splitting (`zsplit`)

Splits a multi-level ROI label into per-level labels (groups of adjacent
z-slices, relabelled 1, 2, …):

```bash
python -m lumbarmri.zsplit split --mask case_01_ME_label.nii.gz --output-dir outputs/levels
```

Output names follow `split_output_name` (documented in
`docs/ALGORITHM_NOTES.md`); use `relabel` / `export` for cleanup and
per-level export.

## Step 5 — BIC-LASSO (`bic_lasso`)

Input is a CSV **table** (rows = subjects, columns = candidate predictors +
one outcome) plus a YAML config that maps raw column names to display names.

1. Copy `configs/lasso_template.yaml` and edit:
   - `outcome`: name of the outcome column in your CSV;
   - `predictors` / `display_names`: your columns (generic example names in
     the template — replace them);
   - keep `seed: 42` for reproducibility.
2. Validate, then run:

```bash
python -m lumbarmri.bic_lasso validate --config my_config.yaml
python -m lumbarmri.bic_lasso run --config my_config.yaml --output outputs/lasso
```

Outputs: Fig. 3 (BIC path + bootstrap band) and Fig. 4 (coefficient paths)
in PNG/PDF/SVG, plus `run_summary.json` with both λ selections
(`lambda_grid_argmin`, LARS), the selected variable set and the λ 95% CI.
Both λ selections must agree on the variable set — the run reports
`check_two_lambdas_agree` as PASS/FAIL. Bootstrap bands describe total
uncertainty (intercept retained): cite `lambda_ci`, not the band width.

To see the whole pipeline on synthetic data first:

```bash
python examples/run_demo.py            # five modules end-to-end, ~15 s
python examples/run_demo.py --quick    # without the LASSO step
```

## Common pitfalls

| symptom | cause / fix |
|---|---|
| `ENOENT` on a path that exists | SimpleITK non-ASCII path bug — keep all IO routed through `lumbarmri.io` (the CLI already does) |
| N4 CV looks worse after correction on a two-tissue phantom | the CV criterion is only valid for single-tissue data; use the Dice criterion there (see `docs/ALGORITHM_NOTES.md` §1) |
| `cv_reduction_pct` sign confusion | in `N4Result` it is `100*(after-before)/before`, so **negative = improvement** |
| ROI statistics disagree with manual means | population std (`ddof=0`) on `mask > 0` only; mask exterior fill must not enter the denominator |
| LASSO raises `assert_standardised` | `run()` standardises internally; if you call the API yourself, feed z-scored inputs |
| filenames not discovered in batch modes | masks must end `_label.nii.gz` (or match `*_label*`) |

## FAQ

**Do I need the same 512×512×12 geometry?** No. All modules read geometry
from the NIfTI headers; only the 2D N4 mode assumes thick-slice anisotropy.

**Where do real study IDs belong?** Nowhere in this repository. Run
`python -m lumbarmri.deidentify` first and keep the mapping file offline
(`docs/ANONYMIZATION.md`).

**Which parts are frozen conventions?** Parameter defaults, seeds and
tolerances are listed in `docs/ALGORITHM_NOTES.md`; `tests/` + per-module
`verify` commands guard them.
