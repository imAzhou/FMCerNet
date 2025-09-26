_base_ = [
    './backbone_cfg.py',
]

net_type = 'patch'
# backbone
backbone_type = 'resnet'
backbone_cfg = _base_.backbone_cfgdict[backbone_type]
backbone_cfg['frozen_backbone'] = False
backbone_cfg['use_peft'] = None

# neck
neck_type = 'identity'

# classifier
taskhead_type = 'cls'
taskhead_model = 'mlc_linear'
positive_thr = 0.5
eval_prime_score = 'multi-label/f1-score'