# SPDX-FileCopyrightText: 2024-present barrettMCW <mjbarrett@mcw.edu>
#
# SPDX-License-Identifier: MIT
"""
Colour preprocessing for H&E whole-slide images.

This module re-exports the lower-level helper functions from
:mod:`shell.transforms` so that existing call-sites (e.g. the OMERO
tile-by-tile pipeline in :mod:`shell.infer_omero_wsi`) continue to work
without modification.

For the full set of MONAI dictionary transforms see
:mod:`shell.transforms`.
"""

from shell.transforms import (  # noqa: F401
    apply_eho_chunked,
    compute_background_intensity,
    detect_background,
    estimate_stain_params,
)