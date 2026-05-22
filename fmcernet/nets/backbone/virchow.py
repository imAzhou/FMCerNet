import torch
from safetensors.torch import load_file
from timm import create_model
from timm.layers import SwiGLUPacked

from .meta_backbone import MetaBackbone


class _VirchowBase(MetaBackbone):
    model_name = None
    num_register_tokens = 0

    def __init__(self, args):
        super(_VirchowBase, self).__init__(args)
        self.backbone = create_model(
            "vit_huge_patch14_224",
            img_size=224,
            patch_size=14,
            init_values=1e-5,
            mlp_ratio=5.3375,
            num_classes=0,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
            dynamic_img_size=True,
            reg_tokens=self.num_register_tokens,
        )
        self.load_backbone(args.backbone_cfg['backbone_ckpt'])
        self.backbone = self.get_peft_model(self.backbone)
        self.freeze_backbone(args.backbone_cfg['frozen_backbone'])

    def load_backbone(self, ckpt):
        params_weight = load_file(ckpt, device='cpu')
        load_result = self.backbone.load_state_dict(params_weight, strict=True)
        print(f'Load backbone {self.model_name}: ' + str(load_result))

    def freeze_backbone(self, frozen_backbone):
        update_keys = ['lora']
        if frozen_backbone:
            for name, param in self.backbone.named_parameters():
                param.requires_grad = False
                for key in update_keys:
                    if key in name:
                        param.requires_grad = True

    def forward(self, x: torch.Tensor):
        output = self.forward_features(x)
        class_token = output[:, 0]
        patch_start = 1 + self.num_register_tokens
        patch_tokens = output[:, patch_start:]
        tile_embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
        output_dict = {
            'x_norm_clstoken': tile_embedding,
            'x_norm_patchtokens': patch_tokens,
            'x_norm_tokens': output,
        }
        if self.num_register_tokens > 0:
            output_dict['x_norm_regtokens'] = output[:, 1:patch_start]
        return output_dict

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.patch_embed(x)
        x = self.backbone._pos_embed(x)
        x = self.backbone.patch_drop(x)
        x = self.backbone.norm_pre(x)
        for blk in self.backbone.blocks:
            x = blk(x)
        x = self.backbone.norm(x)
        return x


class Virchow(_VirchowBase):
    model_name = 'Virchow'
    num_register_tokens = 0


class Virchow2(_VirchowBase):
    model_name = 'Virchow2'
    num_register_tokens = 4
