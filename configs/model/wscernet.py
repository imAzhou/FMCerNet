_base_ = [
    './backbone_cfg.py',
]

# backbone
backbone_type = 'fusionnet'
backbone_cfg = _base_.backbone_cfgdict[backbone_type]

# neck
neck_type = 'identity'

# classifier
taskhead_type = 'cls'
taskhead_model = 'wscer_mlc'
positive_thr = 0.5
# eval_prime_score = 'single-label/binary_accuracy'
# eval_prime_score = 'multi-label/img_accuracy'
eval_prime_score = 'multi-label/f1-score'