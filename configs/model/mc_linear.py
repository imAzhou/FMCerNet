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
taskhead_model = 'mc_linear'
eval_prime_score = 'single-label/mc_f1_score'