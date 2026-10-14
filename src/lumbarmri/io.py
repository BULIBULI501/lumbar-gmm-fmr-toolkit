"""chdir-safe SimpleITK I/O helpers.

Why this module exists
----------------------
SimpleITK cannot open file paths containing non-ASCII characters on some
platforms/builds (it fails with ENOENT even though the file exists).  A
reproducible workaround used throughout this toolkit:

    enter the file's directory with ``chdir()``, then read/write using the
    plain ASCII ``Path.name`` as a relative filename.

``chdir()`` restores the previous working directory on exit, including on
exceptions.  All image reads/writes in this package go through
:func:`read_image` / :func:`write_image`, so the workaround lives in exactly
one place.

.. warning::
   ``os.chdir`` is process-global.  Do not call these helpers concurrently
   from multiple threads.

Import example::

    from lumbarmri.io import read_image, write_image
    img = read_image(Path("case_01_image.nii.gz"))
    write_image(img, Path("out/case_01_n4.nii.gz"))
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Iterator

import SimpleITK as sitk

__all__ = ["chdir", "read_image", "write_image"]


@contextlib.contextmanager
def chdir(path: Path) -> Iterator[None]:
    """Enter ``path`` as working directory for the duration of the block.

    Restores the previous working directory on exit, even on exceptions.
    This is the documented workaround for SimpleITK's ENOENT failure on
    non-ASCII paths: chdir first, then use the plain filename.

    Args:
        path: directory to enter.

    Yields:
        None.
    """
    prev = Path.cwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(str(prev))


def read_image(path: Path) -> sitk.Image:
    """Read a NIfTI/MHA image via the chdir channel.

    Args:
        path: image file path (``.nii.gz`` / ``.nii`` / ``.mha`` ...).

    Returns:
        The SimpleITK image (geometry + pixel data).

    Raises:
        FileNotFoundError: file does not exist.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"image file not found: {path}")
    with chdir(path.parent):
        return sitk.ReadImage(path.name)


def write_image(image: sitk.Image, path: Path) -> Path:
    """Write a SimpleITK image via the chdir channel.

    Parent directories are created on demand.

    Args:
        image: image to write.
        path: output file path (extension selects the format).

    Returns:
        ``path``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with chdir(path.parent):
        sitk.WriteImage(image, path.name)
    return path
