import torch

from .GenBioPathFM.model import VisionTransformer
from .meta_backbone import MetaBackbone


class GenBioPathFM(MetaBackbone):
    def __init__(self, args):
        super(GenBioPathFM, self).__init__(args)
        self.backbone = VisionTransformer(
            img_size=224,
            patch_size=16,
            embed_dim=1536,
            depth=40,
            num_heads=24,
            ffn_ratio=4,
            in_chans=1,
            n_storage_tokens=4,
            ffn_layer='swiglu64',
            layerscale_init=1e-5,
            qkv_bias=False,
            proj_bias=True,
            ffn_bias=True,
            pos_embed_rope_rescale_coords=2,
            pos_embed_rope_jitter_coords=True,
            pos_embed_rope_normalize_coords='separate',
        )
        self.register_buffer('imagenet_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), False)
        self.register_buffer('imagenet_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), False)
        self.register_buffer('pathfm_mean', torch.tensor([0.697, 0.575, 0.728]).view(1, 3, 1, 1), False)
        self.register_buffer('pathfm_std', torch.tensor([0.188, 0.240, 0.187]).view(1, 3, 1, 1), False)
        self.load_backbone(args.backbone_cfg['backbone_ckpt'])
        self.backbone = self.get_peft_model(self.backbone)
        self.freeze_backbone(args.backbone_cfg['frozen_backbone'])

    def load_backbone(self, ckpt):
        params_weight = torch.load(ckpt, map_location='cpu', weights_only=True)
        load_result = self.backbone.load_state_dict(params_weight, strict=True)
        print('Load backbone GenBio-PathFM: ' + str(load_result))

    def freeze_backbone(self, frozen_backbone):
        update_keys = ['lora']
        if frozen_backbone:
            for name, param in self.backbone.named_parameters():
                param.requires_grad = False
                for key in update_keys:
                    if key in name:
                        param.requires_grad = True

    def forward(self, x: torch.Tensor):
        x = self.renormalize_input(x)
        b, c, h, w = x.shape
        features = self.backbone.forward_features(x.reshape(b * c, 1, h, w))

        cls_tokens = features['x_norm_clstoken'].reshape(b, c, -1)
        patch_tokens = features['x_norm_patchtokens']
        num_patches, embed_dim = patch_tokens.shape[1], patch_tokens.shape[2]
        patch_tokens = patch_tokens.reshape(b, c, num_patches, embed_dim)

        cls_embedding = torch.cat([cls_tokens[:, 0], cls_tokens[:, 1], cls_tokens[:, 2]], dim=-1)
        patch_embedding = torch.cat([
            patch_tokens[:, 0],
            patch_tokens[:, 1],
            patch_tokens[:, 2],
        ], dim=-1)
        return {
            'x_norm_clstoken': cls_embedding,
            'x_norm_patchtokens': patch_embedding,
        }

    def renormalize_input(self, x: torch.Tensor):
        x = x * self.imagenet_std + self.imagenet_mean
        x = (x - self.pathfm_mean) / self.pathfm_std
        return x
