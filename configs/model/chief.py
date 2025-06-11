_base_ = [
    './backbone_cfg.py',
]

# backbone
backbone_type = 'uni'
backbone_cfg = _base_.backbone_cfgdict[backbone_type]

# neck
neck_type = None

# classifier
classifier_type = 'chief'
positive_thr = 0.5
eval_prime_score = 'img_accuracy'
