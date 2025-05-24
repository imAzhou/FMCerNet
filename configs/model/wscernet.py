_base_ = [
    './backbone_cfg.py',
]

# backbone
backbone_type = 'uni'
backbone_cfg = _base_.backbone_cfgdict[backbone_type]

# neck
neck_type = 'identity'

# classifier
classifier_type = 'wscer_mlc'
# eval_prime_score = 'single-label/binary_accuracy'
eval_prime_score = 'multi-label/img_accuracy'