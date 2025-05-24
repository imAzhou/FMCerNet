_base_ = [
    './backbone_cfg.py',
]

# backbone
backbone_type = 'sam2'
backbone_cfg = _base_.backbone_cfgdict[backbone_type]

# neck
neck_type = None

# classifier
classifier_type = 'wscer_partial'
binary_branch_input_dim = 256
num_instance_queries = 50
instance_ckpt = '/c22073/zly/codes/sam2/checkpoints/sam2.1_hiera_large.pt'
# instance_ckpt = 'checkpoints/sam2.1_hiera_large.pt'

eval_prime_score = 'img_accuracy'
