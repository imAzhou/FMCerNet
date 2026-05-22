import torch
from mmpretrain import get_model
from .meta_backbone import MetaBackbone

class ResNet(MetaBackbone):
    def __init__(self, args):
        super(ResNet, self).__init__(args)
        self.backbone = get_model(
            'resnet50_8xb32_in1k', 
            pretrained=False, 
            backbone=dict(out_indices=(0, 1, 2, 3))
        ).backbone
        
        backbone_ckpt = args.backbone_cfg['backbone_ckpt']
        if backbone_ckpt is not None:
            self.load_backbone(backbone_ckpt)

    def load_backbone(self, ckpt):
        params_weight = torch.load(ckpt, map_location=self.device)
        new_state_dict = {}
        state_dict = params_weight['state_dict']
        for key,value in state_dict.items():
            new_name = key.replace('backbone.', '')
            new_state_dict[new_name] = value
        load_result = self.backbone.load_state_dict(new_state_dict, strict=False)
        print('Load backbone ResNet50: ' + str(load_result))

    def forward(self, x: torch.Tensor):
        # feature_emb: tuple
        feature_emb = self.backbone(x)
        return feature_emb
