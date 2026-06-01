import torch
import torch.nn as nn
from torch.nn import functional as F
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
        self.feature_fusion = nn.Linear(feat_dim * 2, feat_dim)
        self._init_feature_fusion(feat_dim)
        
        self.load_backbone(args.backbone_cfg['backbone_ckpt'])
        self.vit_module = self.get_peft_model(self.vit_module)
        self.freeze_backbone(args.backbone_cfg['frozen_backbone'])

    def _init_feature_fusion(self, feat_dim):
        layer = self.feature_fusion
        assert layer.weight.shape == (feat_dim, feat_dim * 2)
        nn.init.zeros_(layer.weight)
        nn.init.zeros_(layer.bias)
        with torch.no_grad():
            layer.weight[:, :feat_dim].copy_(
                torch.eye(
                    feat_dim,
                    dtype=layer.weight.dtype,
                    device=layer.weight.device,
                )
            )
            freq_weight = torch.empty(
                feat_dim,
                feat_dim,
                dtype=layer.weight.dtype,
                device=layer.weight.device,
            )
            nn.init.normal_(freq_weight, std=0.001)
            freq_weight.clamp_(min=-0.003, max=0.003)
            layer.weight[:, feat_dim:].copy_(freq_weight)

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
        dtcwt_output = self.dtcwt_module(x) # Tensor: B,N,C

        fused_tokens = self.feature_fusion(
            torch.cat([vit_imgtokens, dtcwt_output], dim=-1)
        )

        output = {
            **vit_output, 
            'dtcwt_output':dtcwt_output,
            'cat_output': fused_tokens,     # Tensor: B,N,C
        }
        return output
