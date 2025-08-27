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
            use_dtcwt_indexes = [],
            dtcwt_featlen = args.input_size // 14
        )
        self.vit_module = vit_large(**vit_kwargs)
        self.dtcwt_module = DTCWTModule(args.input_size)
        
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
        dtcwt_output = self.dtcwt_module(x) # Tensor: B,N,C
        output = {
            **vit_output, 
            'dtcwt_output':dtcwt_output,
        }
        return output

def make_pretrain_ckpt():
    import torch
    from cerwsi.nets.backbone.FusionNet.fusionnet import FusionNet
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.input_size = 1024
    args.backbone_cfg = {
        'use_peft': None,
        'frozen_backbone': True,
        'backbone_ckpt': None
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
    sam_ckpt = torch.load('checkpoints/sam_vit_h_4b8939.pth', map_location="cpu")
    for key,value in sam_ckpt.items():
        if 'patch_embed' in key:
            key = '.'.join(key.split('.')[1:])
            model_ckpt['dtcwt_module.'+key] = value
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
