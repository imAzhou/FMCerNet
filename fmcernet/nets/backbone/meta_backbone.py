import torch
import torch.nn as nn
from peft import LoraConfig, FourierFTConfig, get_peft_model
from abc import abstractmethod

class MetaBackbone(nn.Module):
    def __init__(self, args):
        super(MetaBackbone, self).__init__()
        self.backbone_cfg = args.backbone_cfg
        self.peft_type = self.backbone_cfg['use_peft']

    @property
    def device(self):
        return next(self.parameters()).device        
    
    def get_peft_model(self, model):
        if self.peft_type is None:
            return model
        
        peft_config = None
        if self.peft_type == 'lora':
            target_modules = self.backbone_cfg.get(
                'lora_target_modules',
                ["qkv", "proj", "fc1", "fc2"]
            )
            peft_config = LoraConfig(
                    r=8,  # LoRA 的秩
                    lora_alpha=16,  # LoRA 的缩放因子
                    target_modules = target_modules,  # 应用 LoRA 的目标模块
                    lora_dropout=0.1,  # Dropout 概率
                    bias="none",  # 是否调整偏置
                )
        elif self.peft_type == 'FourierFT':
            target_modules = self.backbone_cfg.get(
                'fourierft_target_modules',
                self.backbone_cfg.get('lora_target_modules', ["qkv", "proj", "fc1", "fc2"])
            )
            peft_config = FourierFTConfig(
                n_frequency = 1000,
                target_modules = target_modules,
                exclude_modules = ["patch_embed.proj"],
                scaling = 300.0
            )
        else:
            raise ValueError(f'Unsupported PEFT type: {self.peft_type}')

        peft_model = get_peft_model(model, peft_config).base_model
        return peft_model
    
    @abstractmethod
    def load_backbone(self, ckpt):
        '''
        Args:
            ckpt (str): backbone checkpoint path
        '''

    @abstractmethod
    def forward(self, x: torch.Tensor):
        '''
        Args:
            x (tensor): input image tensor, shape is (bs, 3, H, W)
        Return:
            featuremap: list(tensor) or tensor, contain CLS token or not.
        '''
