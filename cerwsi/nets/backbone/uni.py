import torch
from timm import create_model
from timm.layers import resample_abs_pos_embed
from .meta_backbone import MetaBackbone


class UNI(MetaBackbone):
    def __init__(self, args):
        super(UNI, self).__init__(args)
        self.backbone = create_model(
            "vit_large_patch16_224", img_size=224, patch_size=16, 
            init_values=1e-5, num_classes=0, dynamic_img_size=True
        )
        self.load_backbone(args.backbone_cfg['backbone_ckpt'])
        self.backbone = self.get_peft_model(self.backbone)
        self.freeze_backbone(args.backbone_cfg['frozen_backbone'])

    def load_backbone(self, ckpt):
        if ckpt is not None:
            params_weight = torch.load(ckpt, map_location='cpu')
            load_result = self.backbone.load_state_dict(params_weight, strict=False)
            print('Load backbone NUI: ' + str(load_result))

    def freeze_backbone(self, frozen_backbone):
        '''frozen the backbone params'''
        update_keys = ['lora']
        if frozen_backbone:
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
        x = self.backbone.patch_drop(x)
        x = self.backbone.norm_pre(x)
        for idx,blk in enumerate(self.backbone.blocks):
            x = blk(x)
        x = self.backbone.norm(x)
        return x
    