"""2D U-Net for CAMUS cardiac segmentation, built from MONAI.

MONAI's ``UNet`` is used rather than a hand-rolled one so the baseline is a
known quantity: residual units, instance norm and the standard encoder/decoder
layout, with nothing bespoke to debug before the data pipeline is trusted.
"""

from __future__ import annotations

from typing import Sequence

import torch
from monai.networks.nets import UNet

from camus_dataset import NUM_CLASSES

#: 5 resolution levels: 256 -> 128 -> 64 -> 32 -> 16 at the bottleneck.
DEFAULT_CHANNELS: tuple[int, ...] = (32, 64, 128, 256, 512)
DEFAULT_STRIDES: tuple[int, ...] = (2, 2, 2, 2)


def build_unet(
    in_channels: int = 1,
    out_channels: int = NUM_CLASSES,
    channels: Sequence[int] = DEFAULT_CHANNELS,
    strides: Sequence[int] = DEFAULT_STRIDES,
    num_res_units: int = 2,
    dropout: float = 0.0,
    norm: str = "INSTANCE",
) -> UNet:
    """Build the 2D U-Net.

    Args:
        in_channels: 1 for B-mode ultrasound.
        out_channels: One logit per class, background included, so the head
            matches ``NUM_CLASSES`` and the loss can use softmax + CE.
        channels: Feature width at each resolution level.
        strides: Downsampling factor between levels; ``len(strides)`` must be
            ``len(channels) - 1``.
        num_res_units: Residual units per level. 2 is MONAI's usual choice and
            trains more stably than plain conv blocks at this depth.
        dropout: Dropout probability inside the conv blocks.
        norm: Normalisation layer. Instance norm is the default rather than
            batch norm because medical segmentation runs at small batch sizes,
            where batch statistics are noisy enough to hurt.

    Returns:
        A MONAI ``UNet`` mapping ``(B, in_channels, H, W)`` to raw logits of
        shape ``(B, out_channels, H, W)``. No activation is applied - the loss
        applies softmax itself.
    """
    if len(strides) != len(channels) - 1:
        raise ValueError(
            f"strides must have len(channels) - 1 = {len(channels) - 1} entries, "
            f"got {len(strides)}."
        )

    return UNet(
        spatial_dims=2,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=tuple(channels),
        strides=tuple(strides),
        num_res_units=num_res_units,
        dropout=dropout,
        norm=norm,
    )


def count_parameters(model: torch.nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def resolve_device(preference: str | None = None) -> torch.device:
    """Pick a device: explicit preference, else CUDA, else MPS, else CPU."""
    if preference:
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


if __name__ == "__main__":
    model = build_unet()
    device = resolve_device()
    model.to(device)

    dummy = torch.randn(2, 1, 256, 256, device=device)
    with torch.inference_mode():
        logits = model(dummy)

    print(f"device      {device}")
    print(f"parameters  {count_parameters(model):,}")
    print(f"input       {tuple(dummy.shape)}")
    print(f"logits      {tuple(logits.shape)}  (expect (2, {NUM_CLASSES}, 256, 256))")
    assert logits.shape == (2, NUM_CLASSES, 256, 256)
    print("forward pass OK")
