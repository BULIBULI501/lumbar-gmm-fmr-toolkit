# Algorithm Notes (frozen conventions)

Every convention below is frozen: the implementation reproduces it exactly,
and `tests/` + the per-module `verify` self-checks guard it.  Do not
"improve" these choices silently — changing any of them changes numbers.

## 1. N4 bias-field correction (`preprocess`)

- Estimation is run on a **downsampled** grid and the field is rebuilt at
  full resolution via `GetLogBiasFieldAsImage`; correction is
  `input / exp(log_bias)` (log-domain bias).
- Two modes: `3d` (whole volume) and `2d` (**per-slice**, recommended for
  anisotropic T2WI, e.g. 512x512x12 with 10-15 mm slice thickness vs
  0.39 mm in-plane).
- Estimation mask: `mask_method='roi'` (default) estimates the field inside
  the ROI so abdominal fat cannot drag the B-spline control points; `'otsu'`,
  `'mean'`, `'percentile'` are available.  Measured on the study data the
  ROI mask reduced inter-side CV variation about 5.6x more than a whole-body
  Otsu mask.
- 2D slices inherit the parent volume's in-plane direction block
  `D[0:2, 0:2]` — an earlier research script dropped directions through a
  dead-code identity check (10/10 measured volumes carry in-plane rotation).
- The SimpleITK non-ASCII path bug (ENOENT on paths containing non-ASCII
  characters) is handled by one shared workaround: `chdir` into the file's
  directory and use the plain ASCII filename (`lumbarmri.io`).  Keep all
  image I/O routed through it.

### Default parameters (`N4Params`)

| parameter | default |
|---|---|
| iterations (per pyramid level) | (50, 50, 50, 50) |
| control points (B-spline) | 4 |
| spline order | 3 |
| bias field FWHM | 0.15 |
| histogram bins | 200 |
| convergence threshold | 0.001 |
| Wiener filter noise | 0.01 |
| shrink factor | 2 |

Note: the historical research scripts used the names `n_control_points`,
`bias_field_fwhm`, `n_histogram_bins`; this package uses the n4tool names
`control_points`, `fwhm`, `histogram_bins`.  Values are identical.

## 2. Intensity normalization (`normalization`)

Four routes, all preserved:

- **ROI z-score** — `ShiftScaleImageFilter` with explicit
  `SetOutputPixelType(sitkFloat32)` (without it an int16 input truncates
  z-scores to integers), mask exterior forced to 0, optional symmetric
  truncation via `ClampImageFilter` at `clip_sigma` (not
  `IntensityWindowing`, which is a display mapping).  Degenerate ROI (zero
  variance): raw gray is kept — distinct from returning zeros.
- **ROI statistics** — mean / population std (`ddof=0`) computed on a
  read-only array view.  Deliberately not SimpleITK statistics operators:
  `LabelStatisticsImageFilter` is unavailable in 3D, and `Mask` + statistics
  counts the outside fill in the denominator (measured: a true ROI mean of
  547.03 came out as 136.76).
- **Nyul standardization** — percentile landmarks on an `arange` grid over
  `[p_low, p_high]` (a `linspace[1:-1]` variant silently produced 9 points
  instead of the requested count), piecewise-linear mapping with
  extrapolation, strictly monotone; models persist as `.npz`.
- **ComBat** — optional extra (`pip install lumbarmri[combat]`), backed by
  the `neuroCombat` package (import name `neuroCombat`).

## 3. Two-component GMM segmentation (`gmm_segmentation`)

Frozen pipeline, in order:

1. ROI intensities = image values where `mask > 0` (not `mask == 1`, so
   repeat-draw masks with value 2 are covered);
2. `StandardScaler`;
3. `GaussianMixture(n_components=2, random_state=42, covariance_type='full')`;
4. **fat cluster = cluster with the larger raw `mean_gray`**;
5. `FM_ratio = V_fat / (V_fat + V_muscle)`, volumes = voxel counts x voxel
   volume.

Degenerate handling: fewer than 2 voxels or a single gray level -> one
cluster, `FM_ratio = 1.0` (reported, not hidden).

Mask naming: `^(?P<stem>.+)_label(?P<draw>\d*)\.nii\.gz$` — `draw` empty =
first annotation round, `2` = repeat draw.

## 4. z-axis splitting (`zsplit`)

- A slice belongs to a z-run when it contains **at least one** target-label
  voxel; empty slices break runs (inclusive endpoints).
- Split outputs keep one slice per file with per-slice origin offset
  `i x spacing[2]` along z.
- Relabel: consecutive runs are relabeled 1, 2, 3, ... in ascending z order.
- Output names: `{base}{draw_token}_{index}{suffix}`; existing files are
  refused by default (`overwrite=False`).

## 5. BIC-LASSO (`bic_lasso`)

- **One traversal over one alpha grid** produces both figures (coefficient
  path + BIC criterion) and therefore exactly **one shared lambda**.
- The criterion is recomputed **point by point**:

  ```
  criterion(a) = n * ln(2*pi*s2) + RSS(a)/s2 + ln(n) * df(a)
  ```

  with `df(a)` = number of non-zero coefficients at alpha `a` and `s2` from
  a single `LassoLarsIC` fit (sklearn's own estimator).  Reading
  `LassoLarsIC.criterion_` directly is wrong: it is defined only at the LARS
  kinks (irregular, decreasing) and cannot fill a regular grid.
- **Dual-lambda disclosure**: the grid argmin and the LARS-kink lambda are
  both reported; they must retain the same variable set
  (`check_two_lambdas_agree`) before either number is quoted.
- **Bootstrap band**: percentile band over resamples with
  `np.random.RandomState(seed)`; the constant term `n*ln(2*pi*s2)` is
  retained, so the band reflects total uncertainty including the noise
  variance estimate.  For stability statements quote `lambda_ci`, never the
  band width.
- Plot style (frozen): `pdf.fonttype=42`, `svg.fonttype='none'`, `tab20`
  colour cycle, line width 1.5, alpha 0.85, x-axis `log10(alpha)` limited to
  `(-3, 1)`, one green vertical line at the BIC optimum, red BIC curve,
  turquoise bootstrap band, all spines visible.
- Predictors must be standardized (`assert_standardised`) — LASSO penalizes
  coefficient magnitude, so unscaled units silently change the selected
  lambda.

## 6. Seed register

| where | seed |
|---|---|
| GMM (`GaussianMixture`) | 42 |
| BIC-LASSO bootstrap (`RandomState`) | 42 |
| Synthetic clinic table (`synth.make_clinic`) | 42 (default) |
| Synthetic phantoms (`preprocess.synthetic_phantom`) | 0 (default) |

## 7. Assertion tolerances

| constant | value | use |
|---|---|---|
| TOL_EXACT | 1e-9 | identical-array / exact recomputation checks |
| TOL_FLOAT | 1e-3 | float quantity comparisons |
| geometry tolerance | 1e-5 | spacing/origin/direction before/after copies |
