import torch
import torch.nn as nn
from torch.nn import functional as F
from peft import LoraConfig, FourierFTConfig, get_peft_model
from ..SmartCCS.vision_transformer import vit_large
from .dtcwt_module import DTCWTModule
from ..meta_backbone import MetaBackbone

def get_peft_config(peft_type:str):
    if peft_type == 'lora':
        return LoraConfig(
                r=8,  # LoRA 的秩
                lora_alpha=16,  # LoRA 的缩放因子
                target_modules = ["attn.qkv", "attn.proj", "lin1", "lin2"],  # 应用 LoRA 的目标模块
                lora_dropout=0.1,  # Dropout 概率
                bias="none",  # 是否调整偏置
            )
    if peft_type == 'FourierFT':
        return FourierFTConfig(
            n_frequency = 1000,
            target_modules = ["qkv", "proj", "fc1", "fc2"],
            exclude_modules = ["patch_embed.proj"],
            scaling = 300.0
        )

class FusionNet(MetaBackbone):
    def __init__(self, args):
        super(FusionNet, self).__init__(args)
        use_peft = args.backbone_cfg['use_peft']
        frozen_backbone = args.backbone_cfg['frozen_backbone']
        backbone_ckpt = args.backbone_cfg['backbone_ckpt']

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
        self.cat_fc = nn.Sequential(
            nn.Linear(2048, 1024),
            # nn.ReLU(),
        )

        if backbone_ckpt is not None:
            self.load_backbone(backbone_ckpt)

        if use_peft in ['lora', 'FourierFT']:
            self.peft_config = get_peft_config(use_peft)
            self.vit_module = get_peft_model(self.vit_module, self.peft_config).base_model

        if frozen_backbone:
            update_keys = ['lora']
            self.freeze_backbone(update_keys)

    def load_backbone(self, ckpt):
        params_weight = torch.load(ckpt, map_location="cpu")
        load_result = self.load_state_dict(params_weight, strict=False)
        print('Load backbone FusionNet: ' + str(load_result))

    def freeze_backbone(self, update_keys):
        '''frozen the backbone params'''
        for name, param in self.vit_module.named_parameters():
            param.requires_grad = False
            for key in update_keys:
                if key in name:
                    param.requires_grad = True

    def forward(self, x: torch.Tensor):
        x_224 = F.interpolate(x,size=224,mode='bilinear',align_corners=False)
        vit_output = self.vit_module(x_224, is_training=True) # dict
        dtcwt_output = self.dtcwt_module(x) # Tensor: B,N,C
        dtcwt_mean = dtcwt_output.mean(dim=1)  # (B, C)
        cls_token_cat = torch.cat([vit_output['x_norm_clstoken'], dtcwt_mean], dim=1)  # (B, 2C)
        cls_token_cat = self.cat_fc(cls_token_cat)
        output = {
            **vit_output, 
            'dtcwt_output':dtcwt_output,
            'cls_token_cat':cls_token_cat,
        }
        return output

def make_pretrain_ckpt(model):
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
