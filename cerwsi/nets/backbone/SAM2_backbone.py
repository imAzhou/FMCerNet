import torch
from peft import LoraConfig, FourierFTConfig, get_peft_model
from types import SimpleNamespace
import yaml
from sam2.modeling.position_encoding import PositionEmbeddingSine
from .SAM2.modeling import ImageEncoder,Hiera,FpnNeck
from .meta_backbone import MetaBackbone

def get_peft_config(peft_type:str):
    if peft_type == 'lora':
        return LoraConfig(
                r=8,  # LoRA 的秩
                lora_alpha=16,  # LoRA 的缩放因子
                target_modules = ["attn.qkv", "attn.proj", "lin1", "lin2"],  # 应用 LoRA 的目标模块
                lora_dropout=0.1,  # Dropout 概率
                bias="none",  # 是否调整偏置
            )
    if peft_type == 'FourierFT':
        return FourierFTConfig(
            n_frequency = 1000,
            target_modules = ["qkv", "proj", "fc1", "fc2"],
            exclude_modules = ["patch_embed.proj"],
            scaling = 300.0
        )

def get_dtcwt_cfg(input_size, trunk_cfg):
    stages = trunk_cfg['stages']
    cur_embed_dim = trunk_cfg['embed_dim']
    downsample_ratio = 2
    featsize,embed_dim = [],[]
    cur_size = input_size // 4  # stage0 之前都做4倍下采样
    for stage_idx, num_blocks in enumerate(stages):
        for _ in range(num_blocks):
            featsize.append(cur_size)
            embed_dim.append(cur_embed_dim)
        cur_size = cur_size // downsample_ratio  # 每个 stage 结束后下采样
        cur_embed_dim = cur_embed_dim * downsample_ratio

    return embed_dim,featsize

class SAM2Encoder(MetaBackbone):
    def __init__(self, args):
        super(SAM2Encoder, self).__init__(args)
        use_peft = args.backbone_cfg['use_peft']
        frozen_backbone = args.backbone_cfg['frozen_backbone']
        use_dtcwt_indexes = args.backbone_cfg['use_dtcwt_indexes']
        backbone_ckpt = args.backbone_cfg['backbone_ckpt']
        backbone_config_file = args.backbone_cfg['config_file']
        with open(backbone_config_file, 'r') as file:
            config = yaml.safe_load(file)

        encoder_cfg = SimpleNamespace(**config['model']['image_encoder'])
        del encoder_cfg.trunk['_target_']
        position_encoding_cfg = encoder_cfg.neck['position_encoding']
        del encoder_cfg.neck['_target_'], encoder_cfg.neck['position_encoding']
        del position_encoding_cfg['_target_']
        dtcwt_embed_dim,dtcwt_featsize = get_dtcwt_cfg(args.input_size, encoder_cfg.trunk)
        self.backbone = ImageEncoder(
            trunk = Hiera(
                **encoder_cfg.trunk,
                use_dtcwt_indexes = use_dtcwt_indexes,
                dtcwt_embed_dim = dtcwt_embed_dim,
                dtcwt_featsize = dtcwt_featsize
            ),
            neck = FpnNeck(
                position_encoding = PositionEmbeddingSine(**position_encoding_cfg),
                **encoder_cfg.neck,
            ),
            scalp = encoder_cfg.scalp
        )

        if backbone_ckpt is not None:
            self.load_backbone(backbone_ckpt)

        if use_peft in ['lora', 'FourierFT']:
            self.peft_config = get_peft_config(use_peft)
            self.backbone = get_peft_model(self.backbone, self.peft_config).base_model

        if frozen_backbone:
            update_keys = ['lora', 'dtxwts']
            self.freeze_backbone(update_keys)

    def load_backbone(self, ckpt):
        params_weight = torch.load(ckpt, map_location="cpu", weights_only=True)["model"]
        state_dict = {}
        for key,value in params_weight.items():
            if 'image_encoder' in key:
                new_name = key.replace('image_encoder.', '')
                state_dict[new_name] = value
        load_result = self.backbone.load_state_dict(state_dict, strict=False)
        print('Load backbone SAM2: ' + str(load_result))

    def freeze_backbone(self, update_keys):
        '''frozen the backbone params'''
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False
            for key in update_keys:
                if key in name:
                    param.requires_grad = True

    def forward(self, x: torch.Tensor):
        dict_output = self.backbone(x)
        return dict_output
