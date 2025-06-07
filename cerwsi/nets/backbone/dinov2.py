import torch
from mmpretrain import get_model
from .meta_backbone import MetaBackbone

class DINOV2(MetaBackbone):
    def __init__(self, args):
        super(DINOV2, self).__init__(args)
        backbone_type = 'dinov2-l'
        backbone_model_name = {
            'dinov2-l': 'vit-large-p14_dinov2-pre_3rdparty'
        }
        self.backbone = get_model(
            backbone_model_name[backbone_type], pretrained=False,
            backbone=dict(out_type='raw', with_cls_token=True)
        ).backbone

    def load_backbone(self, ckpt):
        params_weight = torch.load(ckpt, map_location=self.device)
        new_state_dict = {}
        state_dict = params_weight['state_dict']
        for key,value in state_dict.items():
            new_name = key.replace('backbone.', '')
            new_state_dict[new_name] = value
        print(self.backbone.load_state_dict(new_state_dict, strict=False))
    
    def forward(self, x: torch.Tensor):
        # feature_emb.shape: (bs, 1+num_tokens, C=1024)
        feature_emb = (self.backbone(x))[0]
        return feature_emb
