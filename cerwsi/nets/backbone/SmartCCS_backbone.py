import torch
from torch.nn import functional as F
import math
from peft import LoraConfig, FourierFTConfig, get_peft_model
from .SmartCCS.vision_transformer import vit_large
from .meta_backbone import MetaBackbone

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

class SmartCCS(MetaBackbone):
    def __init__(self, args):
        super(SmartCCS, self).__init__(args)
        use_peft = args.backbone_cfg['use_peft']
        frozen_backbone = args.backbone_cfg['frozen_backbone']
        backbone_ckpt = args.backbone_cfg['backbone_ckpt']
        use_dtcwt_indexes = args.backbone_cfg['use_dtcwt_indexes']

        vit_kwargs = dict(
            img_size=224,
            patch_size=14,
            init_values=1.0e-05,
            ffn_layer='mlp',
            block_chunks=4,
            qkv_bias=True,
            proj_bias=True,
            ffn_bias=True,
            use_dtcwt_indexes = use_dtcwt_indexes,
            dtcwt_featlen = args.input_size // 14
        )
        self.backbone = vit_large(**vit_kwargs)

        if backbone_ckpt is not None:
            self.load_backbone(backbone_ckpt)

        if use_peft in ['lora', 'FourierFT']:
            self.peft_config = get_peft_config(use_peft)
            self.backbone = get_peft_model(self.backbone, self.peft_config).base_model

        if frozen_backbone:
            update_keys = ['lora', 'dtxwts']
            self.freeze_backbone(update_keys)

    def load_backbone(self, ckpt):
        params_weight = torch.load(ckpt, map_location="cpu", weights_only=True)["teacher"]
        state_dict = {}
        for key,value in params_weight.items():
            if 'backbone' in key:
                key = '.'.join(key.split('.')[1:])
                state_dict[key] = value
        load_result = self.backbone.load_state_dict(state_dict, strict=False)
        print('Load backbone SmartCCS: ' + str(load_result))

    def freeze_backbone(self, update_keys):
        '''frozen the backbone params'''
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False
            for key in update_keys:
                if key in name:
                    param.requires_grad = True

    def forward(self, x: torch.Tensor):
        output = self.backbone(x, is_training=True) # dict
        return output
