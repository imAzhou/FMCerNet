import torch
from timm import create_model

from .meta_backbone import MetaBackbone


class GPFM(MetaBackbone):
    def __init__(self, args):
        super(GPFM, self).__init__(args)
        self.backbone = create_model(
            'vit_large_patch14_dinov2',
            pretrained=False,
            img_size=224,
            init_values=1e-5,
        )
        self.load_backbone(args.backbone_cfg['backbone_ckpt'])
        self.backbone = self.get_peft_model(self.backbone)
        self.freeze_backbone(args.backbone_cfg['frozen_backbone'])

    def load_backbone(self, ckpt):
        params_weight = torch.load(ckpt, map_location='cpu', weights_only=True)
        if 'teacher' in params_weight:
            state_dict = params_weight['teacher']
            prefix = self.backbone_cfg['checkpoint_prefix']
            state_dict = {
                key[len(prefix):]: value
                for key, value in state_dict.items()
                if key.startswith(prefix)
            }
        else:
            state_dict = params_weight
        load_result = self.backbone.load_state_dict(state_dict, strict=True)
        print('Load backbone GPFM: ' + str(load_result))

    def freeze_backbone(self, frozen_backbone):
        update_keys = ['lora']
        if frozen_backbone:
            for name, param in self.backbone.named_parameters():
                param.requires_grad = False
                for key in update_keys:
                    if key in name:
                        param.requires_grad = True

    def forward(self, x: torch.Tensor):
        tokens = self.backbone.forward_features(x)
        return {
            'x_norm_clstoken': tokens[:, 0],
            'x_norm_patchtokens': tokens[:, 1:],
            'x_norm_tokens': tokens,
        }
