import torch
from .SmartCCS.vision_transformer import vit_large
from .meta_backbone import MetaBackbone


class SmartCCS(MetaBackbone):
    def __init__(self, args):
        super(SmartCCS, self).__init__(args)
        vit_kwargs = dict(
            img_size=224,
            patch_size=14,
            init_values=1.0e-05,
            ffn_layer='mlp',
            block_chunks=4,
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
            params_weight = torch.load(ckpt, map_location="cpu", weights_only=True)["teacher"]
            state_dict = {}
            for key,value in params_weight.items():
                if 'backbone' in key:
                    key = '.'.join(key.split('.')[1:])
                    state_dict[key] = value
            load_result = self.backbone.load_state_dict(state_dict, strict=False)
            print('Load backbone SmartCCS: ' + str(load_result))

    def freeze_backbone(self, frozen_backbone):
        '''frozen the backbone params'''
        update_keys = ['lora']
        if frozen_backbone:
            for name, param in self.backbone.named_parameters():
                param.requires_grad = False
                # for key in update_keys:
                #     if key in name:
                #         param.requires_grad = True

    def forward(self, x: torch.Tensor):
        output = self.backbone(x, is_training=True) # dict
        return output
