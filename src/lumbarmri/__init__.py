"""lumbarmri -- reproducible lumbar paraspinal MRI toolkit.

Modules:
    preprocess      N4 bias-field correction (3D / per-slice)
    normalization   ROI z-score, Nyul standardization, histogram matching, ComBat
    gmm_segmentation two-component GMM fat/muscle segmentation (FM_ratio)
    zsplit          z-axis split / continuity relabel / group export
    bic_lasso       BIC-LASSO variable selection (config-driven)
    synth           synthetic demo-data generator
    deidentify      NIfTI header scrubbing + filename anonymization
    io              chdir-safe SimpleITK I/O helpers
"""

from .io import chdir, read_image, write_image

__version__ = "0.1.0"

__all__ = ["chdir", "read_image", "write_image", "__version__"]
