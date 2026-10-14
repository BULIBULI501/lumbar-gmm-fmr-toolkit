# lumbar-gmm-fmr-toolkit

Reproducible lumbar paraspinal MRI toolkit (Python 3.9+, SimpleITK).

Five analysis modules plus a de-identification utility, each usable  
standalone or chained:

1. **N4 bias-field correction** (`preprocess`) — 3D and per-slice modes for anisotropic T2WI
2. **Intensity normalization** (`normalization`) — ROI z-score, Nyul standardization, histogram matching, optional ComBat
3. **Two-component GMM segmentation** (`gmm_segmentation`) — fat/muscle clustering, FM_ratio
4. **z-axis mask splitting** (`zsplit`) — per-slice split, z-continuity relabeling, group export
5. **BIC-LASSO variable selection** (`bic_lasso`) — coefficient path + BIC curve + bootstrap stability band
6. **De-identification** (`deidentify`) — metadata/header scrubbing, filename anonymisation, four-check verification

Frozen conventions (parameters, seeds, tolerances) are documented in  
[docs/ALGORITHM\_NOTES.md](docs/ALGORITHM_NOTES.md) and guarded by `tests/`  
plus per-module `verify` self-checks.

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick start

```bash
# full pipeline on bundled synthetic data (~15 s)
python examples/run_demo.py

# same via the console entry point (group -> module CLI)
lumbarmri gmm case --case-dir case_01 --output-dir outputs/gmm
lumbarmri lasso run --config configs/lasso_template.yaml --output outputs/lasso
```

| group  | module           | group    | module    |
| ------ | ---------------- | -------- | --------- |
| `n4`   | preprocess       | `zsplit` | zsplit    |
| `norm` | normalization    | `lasso`  | bic_lasso |
| `gmm`  | gmm_segmentation | `synth`  | synth     |
| `deid` | deidentify       |          |           |

Run `lumbarmri <group> --help` for each group's commands and options.

## Self-checks

```bash
lumbarmri n4 verify
lumbarmri norm verify
lumbarmri gmm verify
lumbarmri zsplit verify
lumbarmri lasso verify
lumbarmri synth verify
lumbarmri deid verify
pytest
```

## Data shipped with this repo

- `examples/demo_data/` — synthetic clinic table (generic column names, seed 42)
- `examples/sample_data/` — three de-identified real cases (image + two ROI  
  labels each); protocol and release record in  
  [docs/ANONYMIZATION.md](docs/ANONYMIZATION.md)

## Documentation

| document                                                          | content                                                   |
| ----------------------------------------------------------------- | --------------------------------------------------------- |
| [docs/ALGORITHM\_NOTES.md](docs/ALGORITHM_NOTES.md)               | frozen conventions, parameter defaults, seeds, tolerances |
| [docs/ANONYMIZATION.md](docs/ANONYMIZATION.md)                    | de-identification protocol, verification record           |
| [docs/ADAPTING\_TO\_YOUR\_DATA.md](docs/ADAPTING_TO_YOUR_DATA.md) | five-step guide for new datasets, pitfalls, FAQ           |
| [docs/PUSH\_CHECKLIST.md](docs/PUSH_CHECKLIST.md)                 | pre-push verification matrix                              |
| [examples/README.md](examples/README.md)                          | demo data provenance and commands                         |

## License

MIT — see [LICENSE](LICENSE).
