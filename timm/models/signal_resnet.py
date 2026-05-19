"""SigResNet: ResNet variants adapted for radar/jamming signal spectrogram classification.

Key adaptations for spectrogram data:
- FreqTimeAttn: dual-axis attention decomposing spatial attention into frequency and time branches
- Attention only in later stages (layer3, layer4) so early layers learn basic features freely
- Deep stem (3x3 convs) to preserve fine spectrogram texture lost by 7x7 convs
- AvgPool downsampling for anti-aliased feature map reduction
"""

from typing import Dict, Optional, Tuple, Type

import torch
import torch.nn as nn

from ._builder import build_model_with_cfg
from ._registry import register_model, generate_default_cfgs
from .resnet import BasicBlock, ResNet


class FreqTimeAttn(nn.Module):
    """Frequency-Time Dual Attention for spectrogram inputs.

    Generates independent attention gates along the frequency (H) and time (W) axes,
    then applies multiplicative fusion. This preserves positional information that
    global-pooling-based attention (SE, ECA) discards.

    - Freq branch: avg pool over time → depthwise 1D conv over frequency → sigmoid
    - Temp branch: avg pool over frequency → depthwise 1D conv over time → sigmoid
    - Output: x * gate_freq * gate_temp

    Args:
        channels: Number of input channels (first positional arg, per create_attn convention).
        freq_kernel_size: Kernel size for the frequency-axis (H) convolution.
        temp_kernel_size: Kernel size for the time-axis (W) convolution.
        gate_layer: Activation function for gating.
    """

    def __init__(
            self,
            channels: int,
            freq_kernel_size: int = 7,
            temp_kernel_size: int = 3,
            gate_layer: Type[nn.Module] = nn.Sigmoid,
            device=None,
            dtype=None,
    ):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()

        # Frequency attention: pool over time axis (W), apply 1D conv over frequency (H)
        pad_freq = freq_kernel_size // 2
        self.conv_freq = nn.Conv2d(
            channels, channels,
            kernel_size=(freq_kernel_size, 1),
            padding=(pad_freq, 0),
            groups=channels,
            bias=False,
            **dd,
        )

        # Time attention: pool over frequency axis (H), apply 1D conv over time (W)
        pad_temp = temp_kernel_size // 2
        self.conv_temp = nn.Conv2d(
            channels, channels,
            kernel_size=(1, temp_kernel_size),
            padding=(0, pad_temp),
            groups=channels,
            bias=False,
            **dd,
        )

        self.gate = gate_layer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Frequency attention: average over time (W) → [N, C, H, 1]
        a_freq = x.mean(dim=3, keepdim=True)
        a_freq = self.conv_freq(a_freq)
        a_freq = self.gate(a_freq)

        # Time attention: average over frequency (H) → [N, C, 1, W]
        a_temp = x.mean(dim=2, keepdim=True)
        a_temp = self.conv_temp(a_temp)
        a_temp = self.gate(a_temp)

        # Multiplicative fusion
        return x * a_freq * a_temp


class SigResNet(ResNet):
    """ResNet variant with FreqTimeAttn applied only to later stages.

    Early stages (layer1, layer2) learn basic features without attention,
    while later stages (layer3, layer4) benefit from frequency-temporal gating
    that captures harmonic patterns and transient structures in spectrograms.
    """

    def __init__(
            self,
            *args,
            attn_stages: Tuple[int, ...] = (3, 4),
            attn_kwargs: Optional[Dict] = None,
            **kwargs,
    ):
        # Strip attn_layer from block_args — we add attention manually after build
        block_args = kwargs.pop('block_args', None) or {}
        block_args.pop('attn_layer', None)

        super().__init__(*args, block_args=block_args, **kwargs)

        # Add FreqTimeAttn only to specified stages
        attn_kwargs = attn_kwargs or {}
        for stage_idx in attn_stages:
            stage = getattr(self, f'layer{stage_idx}')
            for block in stage:
                block.se = FreqTimeAttn(block.conv2.out_channels, **attn_kwargs)


def _create_sigresnet(variant: str, pretrained: bool = False, **kwargs):
    """Create a SigResNet model."""
    return build_model_with_cfg(SigResNet, variant, pretrained, **kwargs)


default_cfgs = generate_default_cfgs({
    'sigresnet18': dict(
        num_classes=1000, input_size=(3, 224, 224), pool_size=(7, 7),
        crop_pct=0.875, interpolation='bilinear',
        first_conv='conv1.0', classifier='fc',
    ),
    'sigresnet34': dict(
        num_classes=1000, input_size=(3, 224, 224), pool_size=(7, 7),
        crop_pct=0.875, interpolation='bilinear',
        first_conv='conv1.0', classifier='fc',
    ),
})


@register_model
def sigresnet18(pretrained: bool = False, **kwargs):
    """SigResNet-18: ResNet-18 adapted for radar signal spectrogram classification.

    FreqTimeAttn is applied to stages 3 and 4 only, allowing early layers to
    learn basic spectrogram features without attention constraints.
    """
    model_args = dict(
        block=BasicBlock,
        layers=(2, 2, 2, 2),
        stem_width=32,
        stem_type='deep',
        avg_down=True,
        attn_stages=(3, 4),
        attn_kwargs=dict(freq_kernel_size=7, temp_kernel_size=3),
    )
    return _create_sigresnet('sigresnet18', pretrained, **dict(model_args, **kwargs))


@register_model
def sigresnet34(pretrained: bool = False, **kwargs):
    """SigResNet-34: ResNet-34 adapted for radar signal spectrogram classification.

    FreqTimeAttn is applied to stages 3 and 4 only, allowing early layers to
    learn basic spectrogram features without attention constraints.
    """
    model_args = dict(
        block=BasicBlock,
        layers=(3, 4, 6, 3),
        stem_width=32,
        stem_type='deep',
        avg_down=True,
        attn_stages=(3, 4),
        attn_kwargs=dict(freq_kernel_size=7, temp_kernel_size=3),
    )
    return _create_sigresnet('sigresnet34', pretrained, **dict(model_args, **kwargs))
