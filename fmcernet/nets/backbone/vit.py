import torch
from mmpretrain import get_model
from .meta_backbone import MetaBackbone

class ViT(MetaBackbone):
    def __init__(self, args):
        super(ViT, self).__init__(args)
        self.backbone = get_model(
            'vit-large-p16_in21k-pre_3rdparty_in1k-384px', 
            pretrained=False,
            backbone=dict(out_type='raw', with_cls_token=True)
        ).backbone
        
        self.load_backbone(args.backbone_cfg['backbone_ckpt'])
        self.backbone = self.get_peft_model(self.backbone)
        self.freeze_backbone(args.backbone_cfg['frozen_backbone'])

    def load_backbone(self, ckpt):
        if ckpt is not None:
            state_dict = torch.load(ckpt, map_location=self.device)
            new_state_dict = {}
            for key,value in state_dict.items():
                new_name = key.replace('backbone.', '')
                new_state_dict[new_name] = value
                
            load_result = self.backbone.load_state_dict(new_state_dict, strict=False)
            print('Load backbone ViT: ' + str(load_result))
    
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
        feature_emb = (self.backbone(x))[0]
        return feature_emb
