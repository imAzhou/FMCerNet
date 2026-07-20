_base_ = [
    './backbone_cfg.py',
]
net_type = 'patch'
# backbone
# backbone_type = 'lfreqvit'
backbone_type = 'lfreqvit'    # smartccs, lfreqvit, cytofm, unicas
backbone_cfg = _base_.backbone_cfgdict[backbone_type]
backbone_cfg['frozen_backbone'] = True
backbone_cfg['use_peft'] = None

# classifier
taskhead_model = 'chief'
positive_thr = 0.5
eval_prime_score = 'accuracy'
