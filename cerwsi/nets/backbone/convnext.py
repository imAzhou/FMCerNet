import torch
from mmpretrain import get_model
from .meta_backbone import MetaBackbone

class ConvNeXt(MetaBackbone):
    def __init__(self, args):
        super(ConvNeXt, self).__init__(args)
        self.backbone = get_model(
            'convnext-large_in21k-pre_3rdparty_in1k', pretrained=False,
            backbone=dict(gap_before_final_norm=False, out_indices=[1,2,3])
        ).backbone
        frozen_backbone = args.backbone_cfg['frozen_backbone']
        backbone_ckpt = args.backbone_cfg['backbone_ckpt']

        if backbone_ckpt is not None:
            self.load_backbone(backbone_ckpt)
        if frozen_backbone:
            self.freeze_backbone()

    def load_backbone(self, ckpt):
        state_dict = (torch.load(ckpt, map_location=self.device))['state_dict']
        new_state_dict = {}
        for key,value in state_dict.items():
            new_name = key.replace('backbone.', '')
            new_state_dict[new_name] = value
        load_result = self.backbone.load_state_dict(new_state_dict, strict=False)
        print('Load backbone ConvNeXt: ' + str(load_result))
    
    def freeze_backbone(self):
        '''frozen the backbone params'''
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor):
        '''
        Args:
            x: (bs, 3, H, W)
        Return:
            dict_output: {
                vision_features: Tensor, (bs, c, h, w)
                backbone_fpn: List[Tensor]: [bs, c, h1,w1]...
            }
            
        '''
        # feature_emb.shape: (bs, C, h, w)
        feature_emb = self.backbone(x)  # 384*64*64，768*32*32，1536*16*16
        return feature_emb
