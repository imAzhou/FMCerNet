import torch
import torch.nn.functional as F
from .SmartCCS.vision_transformer import vit_base
from .meta_backbone import MetaBackbone


class CytoFM(MetaBackbone):
    def __init__(self, args):
        super(CytoFM, self).__init__(args)
        self.input_size = args.backbone_cfg.get('default_input_size', 224)
        vit_kwargs = dict(
            img_size=self.input_size,
            patch_size=args.backbone_cfg.get('vit_patch_size', 16),
            init_values=None,
            ffn_layer='mlp',
            block_chunks=0,
            qkv_bias=True,
            proj_bias=True,
            ffn_bias=True
        )
        self.backbone = vit_base(**vit_kwargs)
        self.load_backbone(args.backbone_cfg['backbone_ckpt'])
        self.backbone = self.get_peft_model(self.backbone)
        self.freeze_backbone(args.backbone_cfg['frozen_backbone'])

    def load_backbone(self, ckpt):
        if ckpt is not None:
            params_weight = torch.load(ckpt, map_location='cpu', weights_only=False)
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
            load_result = self.backbone.load_state_dict(state_dict, strict=False)
            print('Load backbone CytoFM: ' + str(load_result))

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
        output = self.backbone(x, is_training=True) # dict
        return output
