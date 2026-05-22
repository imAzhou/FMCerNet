import torch
from pathlib import Path
from safetensors.torch import load_file
from transformers import AutoModel, DINOv3ConvNextConfig, DINOv3ConvNextModel
from .meta_backbone import MetaBackbone

class ConvNeXt(MetaBackbone):
    def __init__(self, args):
        super(ConvNeXt, self).__init__(args)
        frozen_backbone = args.backbone_cfg['frozen_backbone']
        backbone_ckpt = args.backbone_cfg['backbone_ckpt']
        self.out_indices = tuple(args.backbone_cfg['out_indices'])

        self.load_backbone(backbone_ckpt)
        self.backbone = self.get_peft_model(self.backbone)
        if frozen_backbone:
            self.freeze_backbone()

    def load_backbone(self, ckpt):
        if Path(ckpt).is_file():
            config = DINOv3ConvNextConfig(
                hidden_sizes=[128, 256, 512, 1024],
                depths=[3, 3, 27, 3],
                image_size=224,
            )
            self.backbone = DINOv3ConvNextModel(config)
            load_result = self.backbone.load_state_dict(load_file(ckpt), strict=True)
            print(f'Load backbone ConvNeXt safetensors: {load_result}')
        else:
            self.backbone = AutoModel.from_pretrained(ckpt)
        print(f'Load backbone ConvNeXt from {ckpt} with out_indices={self.out_indices}')
    
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
        outputs = self.backbone(pixel_values=x, output_hidden_states=True)
        return tuple(outputs.hidden_states[index] for index in self.out_indices)
