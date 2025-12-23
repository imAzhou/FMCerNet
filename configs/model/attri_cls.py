_base_ = [
    './backbone_cfg.py',
]

net_type = 'patch'
# backbone
backbone_type = 'smartccs'
backbone_cfg = _base_.backbone_cfgdict[backbone_type]
backbone_cfg['frozen_backbone'] = True
backbone_cfg['use_peft'] = 'lora'

# classifier
taskhead_model = 'attri_cls'
eval_prime_score = 'single-label/mc_f1_score'
load_from = 'log/attri_cls/attribute_classes_sigmoid/2025_12_22_03_03_55/checkpoints/best.pth'