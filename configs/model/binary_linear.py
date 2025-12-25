_base_ = [
    './backbone_cfg.py',
]
net_type = 'patch'
# backbone
backbone_type = 'smartccs'
backbone_cfg = _base_.backbone_cfgdict[backbone_type]

# classifier
taskhead_model = 'binary_linear'
positive_thr = 0.5
eval_prime_score = 'accuracy'
