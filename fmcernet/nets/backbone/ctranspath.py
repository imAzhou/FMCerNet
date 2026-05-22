import torch
from .meta_backbone import MetaBackbone
from .CTransPath import SwinTransformer

class CTransPath(MetaBackbone):
    def __init__(self, args):
        super(CTransPath, self).__init__(args)
        if args.backbone_cfg['use_peft'] is not None:
            raise ValueError('CTransPath does not support PEFT/LoRA mode.')
        self.backbone = SwinTransformer(num_classes=0)
        self.load_backbone(args.backbone_cfg['backbone_ckpt'])
        self.freeze_backbone(args.backbone_cfg['frozen_backbone'])

    def load_backbone(self, ckpt):
        params_weight = torch.load(ckpt, map_location=self.device)
        load_result = self.backbone.load_state_dict(params_weight['model'], strict=True)
        print('Load backbone CTransPath: ' + str(load_result))

    def freeze_backbone(self, frozen_backbone):
        if frozen_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor):
        # feature_emb: (B, 49, 768), the final 7x7 Swin tokens.
        feature_emb = self.backbone.forward_features(x)
        return feature_emb
