import torch
from timm import create_model
from timm.layers import SwiGLUPacked
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
            load_result = self.backbone.load_state_dict(params_weight, strict=True)
            print('Load backbone UNI: ' + str(load_result))

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

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.patch_embed(x)
        x = self.backbone._pos_embed(x)
        x = self.backbone.patch_drop(x)
        x = self.backbone.norm_pre(x)
        for idx,blk in enumerate(self.backbone.blocks):
            x = blk(x)
        x = self.backbone.norm(x)
        return x


class UNI2H(MetaBackbone):
    def __init__(self, args):
        super(UNI2H, self).__init__(args)
        self.backbone = create_model(
            "vit_giant_patch14_224",
            img_size=224,
            patch_size=14,
            depth=24,
            num_heads=24,
            init_values=1e-5,
            embed_dim=1536,
            mlp_ratio=2.66667 * 2,
            num_classes=0,
            no_embed_class=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
            reg_tokens=8,
            dynamic_img_size=True,
        )
        self.load_backbone(args.backbone_cfg['backbone_ckpt'])
        self.backbone = self.get_peft_model(self.backbone)
        self.freeze_backbone(args.backbone_cfg['frozen_backbone'])

    def load_backbone(self, ckpt):
        if ckpt is not None:
            params_weight = torch.load(ckpt, map_location='cpu')
            load_result = self.backbone.load_state_dict(params_weight, strict=True)
            print('Load backbone UNI2-h: ' + str(load_result))

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
        output = self.forward_features(x) # (bs, 1+num_register_tokens+num_tokens, C)
        return output

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.patch_embed(x)
        x = self.backbone._pos_embed(x)
        x = self.backbone.patch_drop(x)
        x = self.backbone.norm_pre(x)
        for idx,blk in enumerate(self.backbone.blocks):
            x = blk(x)
        x = self.backbone.norm(x)
        return x
