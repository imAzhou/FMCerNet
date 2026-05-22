import torch
import torch.nn as nn
import math
from typing import Tuple
from timm.layers import trunc_normal_
from pytorch_wavelets import DTCWTForward

class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class OrientationAttention(nn.Module):
    def __init__(self, channels, num_orientations=6):
        super().__init__()
        self.channels = channels
        self.num_orientations = num_orientations
        self.weight_proj = nn.Conv2d(channels * num_orientations, num_orientations, kernel_size=1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, hf_mag):
        B, C, O, H, W = hf_mag.shape
        assert C == self.channels
        assert O == self.num_orientations
        orientation_input = hf_mag.reshape(B, C * O, H, W)
        orientation_weight = torch.softmax(self.weight_proj(orientation_input), dim=1)
        hf_summary = (hf_mag * orientation_weight.unsqueeze(1)).sum(dim=2)
        return hf_summary


class FrequencyRefineBlock(nn.Module):
    def __init__(self, in_channels, channels):
        super().__init__()
        self.proj_in = nn.Conv2d(in_channels, channels, kernel_size=1)
        self.act1 = nn.GELU()
        self.dwconv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.act2 = nn.GELU()
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
    def forward(self, x):
        shortcut = x
        x = self.proj_in(x)
        x = self.act1(x)
        x = self.dwconv(x)
        x = self.act2(x)
        x = self.proj_out(x)
        if shortcut.shape == x.shape:
            x = x + shortcut
        return x

class TokenProjector(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.norm = nn.LayerNorm(out_channels)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.pool(x)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

class DTCWTModule(nn.Module):
    def __init__(self, input_size, DTBlock_nums):
        super(DTCWTModule, self).__init__()
        patchsize = 16
        embed_dim = 256
        assert input_size % patchsize == 0
        assert DTBlock_nums > 0
        self.patch_embed = PatchEmbed(
            kernel_size=(patchsize, patchsize),
            stride=(patchsize, patchsize),
            in_chans=3,
            embed_dim=embed_dim,
        )
        self.xfm = DTCWTForward(J=2, biort='near_sym_b', qshift='qshift_b')
        self.orientation_attention = OrientationAttention(embed_dim, num_orientations=6)
        encoder_blocks = [FrequencyRefineBlock(embed_dim * 3, embed_dim)]
        encoder_blocks.extend(FrequencyRefineBlock(embed_dim, embed_dim) for _ in range(DTBlock_nums - 1))
        self.frequency_encoder = nn.Sequential(*encoder_blocks)
        self.token_projector = TokenProjector(embed_dim, 1024)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def build_high_frequency_magnitude(self, xh):
        real = xh[..., 0]
        imag = xh[..., 1]
        mag = torch.sqrt(real ** 2 + imag ** 2 + 1e-6)
        return torch.log1p(mag)
    
    def forward(self, x: torch.Tensor):
        x = self.patch_embed(x)
        xl, xh = self.xfm(x)
        ll_feat = xl

        xh0 = xh[0]
        hf_mag = self.build_high_frequency_magnitude(xh0)
        hf_summary = self.orientation_attention(hf_mag)

        xh1 = xh[1]
        hf_mag = self.build_high_frequency_magnitude(xh1)
        hf_summary_l2 = self.orientation_attention(hf_mag)
        hf_summary_l2 = torch.nn.functional.interpolate(
            hf_summary_l2,
            size=hf_summary.shape[-2:],
            mode='bilinear',
            align_corners=False,
        )

        freq_feat = torch.cat([ll_feat, hf_summary, hf_summary_l2], dim=1)
        freq_feat = self.frequency_encoder(freq_feat)
        freq_tokens = self.token_projector(freq_feat)
        return freq_tokens
