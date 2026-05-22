import torch
from pathlib import Path
from safetensors.torch import load_file
from transformers import AutoModel, DINOv3ViTConfig, DINOv3ViTModel
from .meta_backbone import MetaBackbone


class DINOV3(MetaBackbone):
    def __init__(self, args):
        super(DINOV3, self).__init__(args)
        self.load_backbone(args.backbone_cfg['backbone_ckpt'])
        self.num_register_tokens = self.backbone.config.num_register_tokens
        self.backbone = self.get_peft_model(self.backbone)
        self.freeze_backbone(args.backbone_cfg['frozen_backbone'])

    def load_backbone(self, ckpt):
        if Path(ckpt).is_file():
            config = DINOv3ViTConfig(
                hidden_size=1024,
                intermediate_size=4096,
                num_hidden_layers=24,
                num_attention_heads=16,
                patch_size=16,
                num_register_tokens=4,
                image_size=224,
            )
            self.backbone = DINOv3ViTModel(config)
            load_result = self.backbone.load_state_dict(load_file(ckpt), strict=True)
            print(f'Load backbone DINOv3 safetensors: {load_result}')
        else:
            self.backbone = AutoModel.from_pretrained(ckpt)
        print(f'Load backbone DINOv3 from {ckpt}')

    def freeze_backbone(self, frozen_backbone):
        update_keys = ['lora']
        if frozen_backbone:
            for name, param in self.backbone.named_parameters():
                param.requires_grad = False
                for key in update_keys:
                    if key in name:
                        param.requires_grad = True

    def forward(self, x: torch.Tensor):
        outputs = self.backbone(pixel_values=x)
        tokens = outputs.last_hidden_state
        patch_start = 1 + self.num_register_tokens
        return torch.cat([tokens[:, :1, :], tokens[:, patch_start:, :]], dim=1)
