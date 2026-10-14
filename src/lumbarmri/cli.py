"""Thin dispatcher: one ``lumbarmri`` entry point for all module CLIs.

Every module keeps its own ``main(argv)`` and its own ``command`` choices;
this module only maps a short group name to the module and forwards the
remaining arguments unchanged, so nothing is defined twice::

    lumbarmri <group> <command> [options...]

| group    | module            | example |
|----------|-------------------|---------|
| ``n4``     | preprocess      | ``lumbarmri n4 single --image IMG --mask MASK --output-dir OUT`` |
| ``norm``   | normalization   | ``lumbarmri norm zscore --image IMG --mask MASK --output OUT`` |
| ``gmm``    | gmm_segmentation| ``lumbarmri gmm case --case-dir DIR --output-dir OUT`` |
| ``zsplit`` | zsplit          | ``lumbarmri zsplit split --mask MASK --output-dir OUT`` |
| ``lasso``  | bic_lasso       | ``lumbarmri lasso run --config CFG --output OUT`` |
| ``synth``  | synth           | ``lumbarmri synth make --output-dir OUT`` |
| ``deid``   | deidentify      | ``lumbarmri deid run --src SRC --dst DST --map MAP`` |

``python -m lumbarmri.cli`` and the installed console script ``lumbarmri``
are equivalent.  For a group's own help/choices run
``lumbarmri <group> --help`` (each module's ``verify`` command is a
synthetic self-check).
"""

from __future__ import annotations

import sys
from typing import Optional, Sequence

__all__ = ["GROUPS", "main"]

#: group name -> module name inside the package (imported lazily)
GROUPS = {
    "n4": "preprocess",
    "norm": "normalization",
    "gmm": "gmm_segmentation",
    "zsplit": "zsplit",
    "lasso": "bic_lasso",
    "synth": "synth",
    "deid": "deidentify",
}

_USAGE = (
    "usage: lumbarmri <group> <command> [options...]\n"
    "groups: " + ", ".join(sorted(GROUPS)) + "\n"
    "run 'lumbarmri <group> --help' for a group's commands and options"
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Dispatch to the selected module's ``main``; return its exit code."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print(_USAGE)
        return 0
    group, rest = args[0], args[1:]
    module_name = GROUPS.get(group)
    if module_name is None:
        print(f"error: unknown group {group!r}", file=sys.stderr)
        print(_USAGE, file=sys.stderr)
        return 2
    module = __import__(f"lumbarmri.{module_name}", fromlist=["main"])
    return int(module.main(rest))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
