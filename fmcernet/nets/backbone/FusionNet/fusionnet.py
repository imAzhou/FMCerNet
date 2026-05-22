import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from ..SmartCCS.vision_transformer import vit_large
from .dtcwt_module import DTCWTModule
from .feat_pe import PositionEmbeddingRandom
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
            params_weight = torch.load(ckpt, map_location="cpu")
            load_result = self.load_state_dict(params_weight, strict=False)
            print('Load backbone FusionNet: ' + str(load_result))

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

def make_pretrain_ckpt():
    import torch
    from fmcernet.nets.backbone.FusionNet.fusionnet import FusionNet
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.input_size = 1024
    args.backbone_cfg = {
        'use_peft': None,
        'frozen_backbone': True,
        'backbone_ckpt': None,
        'DTBlock_nums': 3
    }

    model = FusionNet(args)
    model_ckpt = {}
    smartccs_ckpt = torch.load('checkpoints/CCS_vitl_100M.pth', map_location="cpu", weights_only=True)["teacher"]
    for key,value in smartccs_ckpt.items():
        if 'backbone' in key:
            key = '.'.join(key.split('.')[1:])
            model_ckpt['vit_module.'+key] = value
    load_result = model.load_state_dict(model_ckpt, strict=False)
    print(load_result)
    torch.save(model.state_dict(), f'checkpoints/fusionnet.pth')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.input_size = 1024
    args.backbone_cfg = {
        'use_peft': 'lora',
        'frozen_backbone': True,
        'backbone_ckpt': None
    }
    data = torch.randn((7, 3, 1024, 1024)).cuda()
    model = FusionNet(args).cuda()
