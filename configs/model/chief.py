_base_ = [
    './backbone_cfg.py',
]
net_type = 'patch'
# backbone
backbone_type = 'fusionnet'
backbone_cfg = _base_.backbone_cfgdict[backbone_type]

# neck
neck_type = None

# classifier
taskhead_type = 'cls'
taskhead_model = 'chief'
positive_thr = 0.5
eval_prime_score = 'accuracy'
