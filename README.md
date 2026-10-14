# lumbar-gmm-fmr-toolkit

Reproducible lumbar paraspinal MRI toolkit (Python 3.9+, SimpleITK).

Five modules, each usable standalone or chained:

1. **N4 bias-field correction** (`preprocess`) — 3D and per-slice modes for anisotropic T2WI
2. **Intensity normalization** (`normalization`) — ROI z-score, Nyul standardization, histogram matching, optional ComBat
3. **Two-component GMM segmentation** (`gmm_segmentation`) — fat/muscle clustering, FM_ratio
4. **z-axis mask splitting** (`zsplit`) — per-slice split, z-continuity relabeling, group export
5. **BIC-LASSO variable selection** (`bic_lasso`) — coefficient path + BIC curve + bootstrap stability band

> Documentation is under active construction; see `docs/` for algorithm notes,
> anonymization protocol, and the "adapting to your data" guide.

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

## Self-checks

```bash
lumbarmri n4 verify
lumbarmri gmm verify
lumbarmri zsplit verify
pytest -q
```

## License

MIT — see [LICENSE](LICENSE).
