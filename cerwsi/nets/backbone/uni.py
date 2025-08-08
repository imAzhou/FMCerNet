import torch
from peft import LoraConfig, FourierFTConfig, get_peft_model
from timm import create_model
from timm.layers import resample_abs_pos_embed
import torch.nn as nn
import math
from .meta_backbone import MetaBackbone
from cerwsi.nets.backbone.PEFT.peft_ours import DTCWTModule

def get_peft_config(peft_type:str):
    if peft_type == 'lora':
        return LoraConfig(
                r=8,  # LoRA 的秩
                lora_alpha=16,  # LoRA 的缩放因子
                target_modules = ["qkv", "proj", "fc1", "fc2"],  # 应用 LoRA 的目标模块
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

class UNI(MetaBackbone):
    def __init__(self, args):
        super(UNI, self).__init__(args)
        
        backbone_ckpt = args.backbone_cfg['backbone_ckpt']
        use_peft = args.backbone_cfg['use_peft']
        frozen_backbone = args.backbone_cfg['frozen_backbone']
        use_dtcwt_indexes = args.backbone_cfg['use_dtcwt_indexes']
        
        self.backbone = create_model(
            "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
        )

        if backbone_ckpt is not None:
            self.load_backbone(backbone_ckpt)

        if use_peft is not None:
            self.peft_config = get_peft_config(use_peft)
            self.backbone = get_peft_model(self.backbone, self.peft_config).base_model

        self.use_dtcwt_indexes = use_dtcwt_indexes
        if len(use_dtcwt_indexes) > 0:
            embed_dim = args.backbone_cfg['backbone_output_dim'][-1]
            dtcwt_featlen = args.input_size // 16
            self.backbone.dtxwts = nn.ModuleList()
            for i in use_dtcwt_indexes:
                self.backbone.dtxwts.append(DTCWTModule(
                    dim=embed_dim,
                    feat_size=dtcwt_featlen
                ))
        
        if frozen_backbone:
            update_keys = ['lora', 'dtxwts']
            self.freeze_backbone(update_keys)

    def load_backbone(self, ckpt):
        params_weight = torch.load(ckpt, map_location='cpu')
        load_result = self.backbone.load_state_dict(params_weight, strict=False)
        print('Load backbone NUI: ' + str(load_result))

    def freeze_backbone(self, update_keys):
        '''frozen the backbone params'''
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False
            for key in update_keys:
                if key in name:
                    param.requires_grad = True

    def forward(self, x: torch.Tensor):
        output = self.forward_features(x) # (bs, 1+num_tokens, C)
        return output
    
    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        prev_grid_size = self.backbone.model.patch_embed.grid_size
        pos_embed = resample_abs_pos_embed(
            self.pos_embed,
            new_size=(H, W),
            old_size=prev_grid_size,
            num_prefix_tokens=self.num_classes,
        )
        x = x.view(B, -1, C)
        to_cat = []
        to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        x = torch.cat(to_cat + [x], dim=1)
        x = x + pos_embed
        return x

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.patch_embed(x)
        x = self.backbone._pos_embed(x)
        if len(self.use_dtcwt_indexes) > 0:
            B,num_tokens,C = x.shape
            featlen = int(math.sqrt(num_tokens-1))
            # B, H, W, C = input_x.shape
            input_x = x[:,1:,:].reshape(B,featlen,featlen,C)
            output_x = self.backbone.dtxwts[0](input_x)
            output_x = output_x.reshape(B,-1,C)
            x = torch.cat(( x[:,0,:].unsqueeze(1), output_x), dim=1)

        x = self.backbone.patch_drop(x)
        x = self.backbone.norm_pre(x)
        for idx,blk in enumerate(self.backbone.blocks):
            x = blk(x)
            if idx+1 in self.use_dtcwt_indexes:
                B,num_tokens,C = x.shape
                featlen = int(math.sqrt(num_tokens-1))
                # B, H, W, C = input_x.shape
                input_x = x[:,1:,:].reshape(B,featlen,featlen,C)
                output_x = self.backbone.dtxwts[idx+1](input_x)
                output_x = output_x.reshape(B,-1,C)
                x = torch.cat(( x[:,0,:].unsqueeze(1), output_x), dim=1)

        x = self.backbone.norm(x)
        return x
    