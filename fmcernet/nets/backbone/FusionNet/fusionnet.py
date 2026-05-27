import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np
from typing import Any, Optional, Tuple
from ..SmartCCS.vision_transformer import vit_large
from .dtcwt_module import DTCWTModule
from ..meta_backbone import MetaBackbone

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class ConcatFCFusion(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.feat_dim = feat_dim
        self.token_fusion = nn.Linear(feat_dim * 2, feat_dim)
        self.cls_fusion = nn.Linear(feat_dim * 2, feat_dim)
        self._init_identity_fusion()

    def _init_identity_fusion(self):
        for layer in [self.token_fusion, self.cls_fusion]:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
            with torch.no_grad():
                layer.weight[:, :self.feat_dim].copy_(
                    torch.eye(
                        self.feat_dim,
                        dtype=layer.weight.dtype,
                        device=layer.weight.device,
                    )
                )
                freq_weight = torch.empty(
                    self.feat_dim,
                    self.feat_dim,
                    dtype=layer.weight.dtype,
                    device=layer.weight.device,
                )
                nn.init.normal_(freq_weight, std=0.001)
                freq_weight.clamp_(min=-0.003, max=0.003)
                layer.weight[:, self.feat_dim:].copy_(freq_weight)

    def forward(self, vit_tokens, dtcwt_tokens, vit_cls):
        assert vit_tokens.shape == dtcwt_tokens.shape, \
            f"Expected matched token shapes, got {tuple(vit_tokens.shape)} and {tuple(dtcwt_tokens.shape)}."
        assert vit_cls.shape == vit_tokens[:, 0].shape, \
            f"Expected cls shape {tuple(vit_tokens[:, 0].shape)}, got {tuple(vit_cls.shape)}."

        freq_cls = dtcwt_tokens.mean(dim=1)
        fused_tokens = self.token_fusion(torch.cat([vit_tokens, dtcwt_tokens], dim=-1))
        fused_cls = self.cls_fusion(torch.cat([vit_cls, freq_cls], dim=-1))
        return fused_tokens, fused_cls

class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        # x_embed、y_embed：每个网格点相对于特征宽高的相对位置
        y_embed = y_embed / h
        x_embed = x_embed / w
        
        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]   #W
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]   #H
        return self._pe_encoding(coords.to(torch.float))  # B x N x C


class FusionNet(MetaBackbone):
    def __init__(self, args):
        super(FusionNet, self).__init__(args)
        vit_kwargs = dict(
            img_size=224,
            patch_size=14,
            init_values=1.0e-05,
            ffn_layer='mlp',
            block_chunks=4,
            qkv_bias=True,
            proj_bias=True,
            ffn_bias=True,
        )
        self.vit_module = vit_large(**vit_kwargs)
        self.dtcwt_module = DTCWTModule(args.input_size, args.backbone_cfg['DTBlock_nums'])
        feat_dim = 1024
        self.feature_fusion = ConcatFCFusion(feat_dim)
        self.pe_layer = PositionEmbeddingRandom(feat_dim // 2)
        
        self.load_backbone(args.backbone_cfg['backbone_ckpt'])
        self.vit_module = self.get_peft_model(self.vit_module)
        self.freeze_backbone(args.backbone_cfg['frozen_backbone'])

    def load_backbone(self, ckpt):
        if ckpt is not None:
            params_weight = torch.load(ckpt, map_location="cpu", weights_only=True)["teacher"]
            state_dict = {}
            for key, value in params_weight.items():
                if key.startswith('backbone.'):
                    state_dict[key[len('backbone.'):]] = value
            self.vit_module.load_state_dict(state_dict, strict=True)
            print('Load vit_module FusionNet from: ' + str(ckpt))

    def freeze_backbone(self, frozen_backbone):
        '''frozen the vit_module params'''
        update_keys = ['lora']
        if frozen_backbone:
            for name, param in self.vit_module.named_parameters():
                param.requires_grad = False
                for key in update_keys:
                    if key in name:
                        param.requires_grad = True

    def forward(self, x: torch.Tensor):
        x_224 = F.interpolate(x,size=224,mode='bilinear',align_corners=False)
        vit_output = self.vit_module(x_224, is_training=True) # dict
        vit_imgtokens = vit_output['x_norm_patchtokens'] # Tensor: B,N,C
        feat_size = int(math.sqrt(vit_imgtokens.shape[1]))
        feat_pe = self.pe_layer((feat_size,feat_size)).unsqueeze(0) # (1,C,H,W)
        feat_pe = feat_pe.flatten(2).permute(0, 2, 1).to(self.device) #  (1, N, C)

        vit_tokens = vit_imgtokens + feat_pe
        dtcwt_output = self.dtcwt_module(x) # Tensor: B,N,C
        img_token_cat, fusion_cls = self.feature_fusion(
            vit_tokens,
            dtcwt_output,
            vit_output['x_norm_clstoken'],
        )

        output = {
            **vit_output, 
            'dtcwt_output':dtcwt_output,
            'cat_output': img_token_cat,     # Tensor: B,N,C
            'fusion_clstoken': fusion_cls,
        }
        return output
