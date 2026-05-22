import torch
import torch.nn.functional as F
from .SmartCCS.vision_transformer import vit_large
from .meta_backbone import MetaBackbone


class UniCAS(MetaBackbone):
    def __init__(self, args):
        super(UniCAS, self).__init__(args)
        self.input_size = args.backbone_cfg.get('default_input_size', 224)
        vit_kwargs = dict(
            img_size=self.input_size,
            patch_size=args.backbone_cfg.get('vit_patch_size', 16),
            init_values=1.0e-05,
            ffn_layer='swiglufused',
            block_chunks=0,
            qkv_bias=True,
            proj_bias=True,
            ffn_bias=True
        )
        self.backbone = vit_large(**vit_kwargs)
        self.load_backbone(args.backbone_cfg['backbone_ckpt'])
        self.backbone = self.get_peft_model(self.backbone)
        self.freeze_backbone(args.backbone_cfg['frozen_backbone'])

    def load_backbone(self, ckpt):
        if ckpt is not None:
            params_weight = torch.load(ckpt, map_location='cpu', weights_only=False)
            params_weight = params_weight.get('state_dict', params_weight)
            params_weight = params_weight.get('model', params_weight)
            params_weight = params_weight.get('teacher', params_weight)
            state_dict = {}
            for key, value in params_weight.items():
                if key.startswith('module.backbone.'):
                    state_dict[key[len('module.backbone.'):]] = value
                elif key.startswith('backbone.'):
                    state_dict[key[len('backbone.'):]] = value
                elif key.startswith('module.'):
                    state_dict[key[len('module.'):]] = value
                else:
                    state_dict[key] = value
            state_dict = {
                key.replace('.mlp.fc1.', '.mlp.w12.').replace('.mlp.fc2.', '.mlp.w3.'): value
                for key, value in state_dict.items()
            }
            load_result = self.backbone.load_state_dict(state_dict, strict=False)
            print('Load backbone UniCAS: ' + str(load_result))

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
        if x.shape[-2:] != (self.input_size, self.input_size):
            x = F.interpolate(
                x,
                size=(self.input_size, self.input_size),
                mode='bilinear',
                align_corners=False,
            )
        output = self.backbone(x, is_training=True) # dict
        return output
