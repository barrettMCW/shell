# SPDX-FileCopyrightText: 2024-present barrettMCW <mjbarrett@mcw.edu>
#
# SPDX-License-Identifier: MIT
"""
Sliding-window inference for whole-slide EHO images.
"""

from __future__ import annotations

import gc
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from monai.inferers.utils import sliding_window_inference
from torch.amp import autocast

# ---------------------------------------------------------------------------
# Default inference parameters
# ---------------------------------------------------------------------------
VAL_ROI_SIZE: tuple[int, int] = (512, 512)
VAL_SW_BATCH: int = 16
VAL_SW_OVERLAP: float = 0.25


def run_inference(
    eho_image: np.ndarray,
    model: torch.nn.Module,
    device: torch.device | str = "cpu",
    *,
    roi_size: tuple[int, int] = VAL_ROI_SIZE,
    sw_batch_size: int = VAL_SW_BATCH,
    overlap: float = VAL_SW_OVERLAP,
) -> np.ndarray:
    """Run sliding-window inference on an EHO (H, W, 3) uint8 image.

    :param eho_image: (H, W, 3) uint8 EHO image.
    :param model: trained SegResNetVAE in eval mode.
    :param device: computation device.
    :param roi_size: sliding-window patch size.
    :param sw_batch_size: number of patches per forward pass.
    :param overlap: fraction of overlap between sliding-window patches.
        Values > 0 enable Gaussian importance weighting so that
        overlapping patch centres contribute more than edges, which
        eliminates the grid artefacts visible with ``mode="constant"``.
    :return: (H, W) uint8 label map.
    """
    device_obj = torch.device(device) if isinstance(device, str) else device

    # HWC uint8 → NCHW float32 [0, 1]
    img_t = (
        torch.from_numpy(eho_image).permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
    )
    del eho_image

    # Pad to multiple of 64
    h, w = img_t.shape[-2:]
    pad_h = (64 - h % 64) % 64
    pad_w = (64 - w % 64) % 64
    padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
    if pad_h or pad_w:
        img_t = F.pad(img_t, padding, "constant", 0)

    amp_device = "cuda" if device_obj.type == "cuda" else "cpu"
    with (
        torch.inference_mode(),
        autocast(amp_device),
        warnings.catch_warnings(),
    ):
        # Suppress MONAI's use of deprecated non-tuple sequence indexing
        # (fixed upstream but not yet released).
        warnings.filterwarnings(
            "ignore",
            message="Using a non-tuple sequence for multidimensional indexing",
            category=UserWarning,
        )
        # Use Gaussian importance weighting when patches overlap so that
        # each patch's centre is trusted more than its edges.  With
        # overlap=0 fall back to uniform ("constant") weighting.
        blend_mode = "gaussian" if overlap > 0 else "constant"
        logits = sliding_window_inference(
            img_t,
            roi_size,
            sw_batch_size,
            model,
            overlap=overlap,
            sw_device=device_obj,
            device=torch.device("cpu"),
            mode=blend_mode,
        )
    del img_t

    # Remove padding
    if pad_h or pad_w:
        _, _, ph, pw = logits.shape
        logits = logits[
            :, :, padding[2] : ph - padding[3], padding[0] : pw - padding[1]
        ]

    pred = torch.argmax(logits[0], dim=0).numpy().astype(np.uint8)
    del logits
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return pred
