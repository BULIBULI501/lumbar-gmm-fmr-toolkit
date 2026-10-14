"""Thin wrappers: every module's ``self_check()`` must pass all records.

Each module carries its own synthetic self-check (runnable via
``python -m lumbarmri.<module> verify``).  This file locks them into pytest
so a single ``pytest -q`` covers the full self-check matrix.
"""

import pytest

from lumbarmri import (
    bic_lasso,
    deidentify,
    gmm_segmentation,
    normalization,
    preprocess,
    synth,
    zsplit,
)

MODULES = [preprocess, normalization, gmm_segmentation, zsplit, bic_lasso, synth, deidentify]


@pytest.mark.parametrize("mod", MODULES, ids=lambda m: m.__name__)
def test_self_check_all_records_pass(mod):
    records = mod.self_check()
    assert len(records) >= 4, f"{mod.__name__}.self_check returned too few records"
    failed = [(name, msg) for name, ok, msg in records if not ok]
    assert not failed, f"{mod.__name__} self_check failures: {failed}"
