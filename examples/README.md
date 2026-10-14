# Examples

## What ships with this repository

### `sample_data/` — de-identified real MRIs (3 cases)

Three lumbar paraspinal T2WI cases from the study cohort, fully
de-identified before commit:

| file pattern | content |
|---|---|
| `case_NN_image.nii.gz` | 512x512x12 T2WI volume (float32) |
| `case_NN_ME_label.nii.gz` | multifidus–erector spinae complex ROI (uint8) |
| `case_NN_P_label.nii.gz` | psoas major ROI (uint8) |

- Names are sequential labels only; the real-ID <-> case mapping is kept
  **off-repository** by design.
- Headers were scrubbed (metadata whitelist + NIfTI-1 text-field zeroing)
  and pixel data verified bit-identical before/after.  Protocol, field
  tables and verification records: [`docs/ANONYMIZATION.md`](../docs/ANONYMIZATION.md).
- Public sharing of these three cases is authorised (ethics / data-sharing
  confirmation obtained before inclusion).

### `demo_data/` — synthetic clinic table (no real patients)

`demo_clinic.csv` (200 rows x 18 generic predictors `var_01..var_18` + `y`)
with four planted signal coefficients; `synth_truth.json` records the
truth, `demo_baseline.json` is the committed reproducibility anchor.  All
values are synthetic — regenerate deterministically with
`make_demo_clinic.py` (seed 42).

## Demo commands

```bash
# five modules end-to-end (N4 -> z-score -> GMM -> zsplit -> BIC-LASSO)
python examples/run_demo.py

# quick variant without the LASSO step
python examples/run_demo.py --quick

# regenerate the synthetic clinic table (deterministic)
python examples/make_demo_clinic.py

# generate purely synthetic NIfTI samples (e.g. when you cannot share MRI)
python examples/make_sample_data.py --output-dir ../outputs/synth_samples
```

All demo products are written to `outputs/demo_run/` (git-ignored).

## Using your own data

Follow [`docs/ADAPTING_TO_YOUR_DATA.md`](../docs/ADAPTING_TO_YOUR_DATA.md):
keep the `case_NN_{image,ME_label,P_label}.nii.gz` naming (mask suffix
`_label`, optional repeat-draw `_label2`), copy
`configs/lasso_template.yaml`, replace the column names, run
`validate` then `run`.
