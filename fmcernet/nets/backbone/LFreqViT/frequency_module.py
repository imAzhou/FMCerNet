import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DTCWTForward, DWTForward
from timm.layers import trunc_normal_


FREQUENCY_CHANNELS = 256
FREQUENCY_INPUT_SIZE = 64
FREQUENCY_OUTPUT_SIZE = 32


def _validate_feature_map(x: torch.Tensor, spatial_size: int) -> None:
    expected = (FREQUENCY_CHANNELS, spatial_size, spatial_size)
    if x.ndim != 4 or tuple(x.shape[1:]) != expected:
        raise ValueError(
            f"Expected frequency feature map with shape (B, {expected[0]}, "
            f"{expected[1]}, {expected[2]}), got {tuple(x.shape)}."
        )


class PatchEmbed(nn.Module):
    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = FREQUENCY_CHANNELS,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class OrientationAttention(nn.Module):
    def __init__(self, channels: int, num_orientations: int) -> None:
        super().__init__()
        self.channels = channels
        self.num_orientations = num_orientations
        self.weight_proj = nn.Conv2d(
            channels * num_orientations,
            num_orientations,
            kernel_size=1,
        )

    def forward(self, high_frequency: torch.Tensor) -> torch.Tensor:
        if high_frequency.ndim != 5:
            raise ValueError(
                "OrientationAttention expects a five-dimensional tensor, "
                f"got shape {tuple(high_frequency.shape)}."
            )
        batch_size, channels, orientations, height, width = high_frequency.shape
        if channels != self.channels or orientations != self.num_orientations:
            raise ValueError(
                f"Expected (C, O)=({self.channels}, {self.num_orientations}), "
                f"got ({channels}, {orientations})."
            )
        attention_input = high_frequency.reshape(
            batch_size,
            channels * orientations,
            height,
            width,
        )
        weights = torch.softmax(self.weight_proj(attention_input), dim=1)
        return (high_frequency * weights.unsqueeze(1)).sum(dim=2)


class FrequencyRefineBlock(nn.Module):
    def __init__(self, in_channels: int, channels: int) -> None:
        super().__init__()
        self.proj_in = nn.Conv2d(in_channels, channels, kernel_size=1)
        self.act1 = nn.GELU()
        self.dwconv = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
        )
        self.act2 = nn.GELU()
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.proj_in(x)
        x = self.act1(x)
        x = self.dwconv(x)
        x = self.act2(x)
        x = self.proj_out(x)
        if shortcut.shape == x.shape:
            x = x + shortcut
        return x


def _build_frequency_encoder(block_count: int) -> nn.Sequential:
    if block_count <= 0:
        raise ValueError(f"DTBlock_nums must be positive, got {block_count}.")
    blocks = [FrequencyRefineBlock(FREQUENCY_CHANNELS * 3, FREQUENCY_CHANNELS)]
    blocks.extend(
        FrequencyRefineBlock(FREQUENCY_CHANNELS, FREQUENCY_CHANNELS)
        for _ in range(block_count - 1)
    )
    return nn.Sequential(*blocks)


class DTCWTFrequencyOperator(nn.Module):
    def __init__(self, block_count: int) -> None:
        super().__init__()
        self.transform = DTCWTForward(
            J=2,
            biort="near_sym_b",
            qshift="qshift_b",
        )
        self.orientation_attention = OrientationAttention(
            FREQUENCY_CHANNELS,
            num_orientations=6,
        )
        self.frequency_encoder = _build_frequency_encoder(block_count)

    @staticmethod
    def _complex_magnitude(coefficients: torch.Tensor) -> torch.Tensor:
        real = coefficients[..., 0]
        imaginary = coefficients[..., 1]
        magnitude = torch.sqrt(real.square() + imaginary.square() + 1e-6)
        return torch.log1p(magnitude)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _validate_feature_map(x, FREQUENCY_INPUT_SIZE)
        low_frequency, high_frequencies = self.transform(x)
        level_one = self.orientation_attention(
            self._complex_magnitude(high_frequencies[0])
        )
        level_two = self.orientation_attention(
            self._complex_magnitude(high_frequencies[1])
        )
        level_two = F.interpolate(
            level_two,
            size=(FREQUENCY_OUTPUT_SIZE, FREQUENCY_OUTPUT_SIZE),
            mode="bilinear",
            align_corners=False,
        )
        features = torch.cat([low_frequency, level_one, level_two], dim=1)
        output = self.frequency_encoder(features)
        _validate_feature_map(output, FREQUENCY_OUTPUT_SIZE)
        return output


class DWTHaarFrequencyOperator(nn.Module):
    def __init__(self, block_count: int) -> None:
        super().__init__()
        self.transform = DWTForward(J=2, wave="haar", mode="zero")
        self.orientation_attention = OrientationAttention(
            FREQUENCY_CHANNELS,
            num_orientations=3,
        )
        self.frequency_encoder = _build_frequency_encoder(block_count)

    def _summarize_high_frequency(self, coefficients: torch.Tensor) -> torch.Tensor:
        magnitude = torch.log1p(torch.abs(coefficients))
        return self.orientation_attention(magnitude)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _validate_feature_map(x, FREQUENCY_INPUT_SIZE)
        low_frequency, high_frequencies = self.transform(x)
        low_frequency = F.interpolate(
            low_frequency,
            size=(FREQUENCY_OUTPUT_SIZE, FREQUENCY_OUTPUT_SIZE),
            mode="bilinear",
            align_corners=False,
        )
        level_one = self._summarize_high_frequency(high_frequencies[0])
        level_two = self._summarize_high_frequency(high_frequencies[1])
        level_two = F.interpolate(
            level_two,
            size=(FREQUENCY_OUTPUT_SIZE, FREQUENCY_OUTPUT_SIZE),
            mode="bilinear",
            align_corners=False,
        )
        features = torch.cat([low_frequency, level_one, level_two], dim=1)
        output = self.frequency_encoder(features)
        _validate_feature_map(output, FREQUENCY_OUTPUT_SIZE)
        return output


class FFTRadialBandsFrequencyOperator(nn.Module):
    def __init__(self, block_count: int) -> None:
        super().__init__()
        frequencies = torch.fft.fftshift(torch.fft.fftfreq(FREQUENCY_INPUT_SIZE))
        vertical, horizontal = torch.meshgrid(frequencies, frequencies, indexing="ij")
        radius = torch.sqrt(horizontal.square() + vertical.square())
        radius = radius / radius.max()
        low_mask = radius < (1.0 / 3.0)
        middle_mask = (radius >= (1.0 / 3.0)) & (radius < (2.0 / 3.0))
        high_mask = radius >= (2.0 / 3.0)
        self.register_buffer("low_mask", low_mask[None, None], persistent=True)
        self.register_buffer("middle_mask", middle_mask[None, None], persistent=True)
        self.register_buffer("high_mask", high_mask[None, None], persistent=True)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.frequency_encoder = _build_frequency_encoder(block_count)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _validate_feature_map(x, FREQUENCY_INPUT_SIZE)
        spectrum = torch.fft.fftshift(
            torch.fft.fft2(x, norm="ortho"),
            dim=(-2, -1),
        )
        magnitude = torch.log1p(torch.abs(spectrum))
        bands = [
            self.pool(magnitude * self.low_mask),
            self.pool(magnitude * self.middle_mask),
            self.pool(magnitude * self.high_mask),
        ]
        features = torch.cat(bands, dim=1)
        output = self.frequency_encoder(features)
        _validate_feature_map(output, FREQUENCY_OUTPUT_SIZE)
        return output


class TokenProjector(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _validate_feature_map(x, FREQUENCY_OUTPUT_SIZE)
        x = self.pool(x)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x)


class FrequencyModule(nn.Module):
    SUPPORTED_OPERATORS = ("dtcwt", "dwt_haar", "fft_radial")

    def __init__(self, input_size: int, block_count: int, operator_name: str) -> None:
        super().__init__()
        if input_size != 1024:
            raise ValueError(
                "LFreqViT frequency experiments require input_size=1024, "
                f"got {input_size}."
            )
        self.patch_embed = PatchEmbed()
        if operator_name == "dtcwt":
            self.frequency_operator = DTCWTFrequencyOperator(block_count)
        elif operator_name == "dwt_haar":
            self.frequency_operator = DWTHaarFrequencyOperator(block_count)
        elif operator_name == "fft_radial":
            self.frequency_operator = FFTRadialBandsFrequencyOperator(block_count)
        else:
            raise ValueError(
                f"Unsupported frequency_operator={operator_name!r}; "
                f"expected one of {self.SUPPORTED_OPERATORS}."
            )
        self.token_projector = TokenProjector(FREQUENCY_CHANNELS, 1024)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Conv2d):
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                module.bias.data.zero_()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        expected = (3, 1024, 1024)
        if x.ndim != 4 or tuple(x.shape[1:]) != expected:
            raise ValueError(
                f"Expected LFreqViT image input with shape (B, 3, 1024, 1024), "
                f"got {tuple(x.shape)}."
            )
        embedded = self.patch_embed(x)
        _validate_feature_map(embedded, FREQUENCY_INPUT_SIZE)
        return self.frequency_operator(embedded)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        tokens = self.token_projector(features)
        expected = (256, 1024)
        if tokens.ndim != 3 or tuple(tokens.shape[1:]) != expected:
            raise ValueError(
                f"Expected frequency tokens with shape (B, 256, 1024), "
                f"got {tuple(tokens.shape)}."
            )
        return tokens
