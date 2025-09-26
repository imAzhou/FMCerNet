_base_ = [
    './backbone_cfg.py',
]

net_type = 'patch'
# backbone
backbone_type = 'fusionnet'
backbone_cfg = _base_.backbone_cfgdict[backbone_type]
backbone_cfg['frozen_backbone'] = True
backbone_cfg['use_peft'] = 'lora'

# neck
neck_type = 'identity'

# classifier
taskhead_type = 'cls'
taskhead_model = 'wscer_mlc'
positive_thr = 0.5
eval_prime_score = 'multi-label/f1-score'