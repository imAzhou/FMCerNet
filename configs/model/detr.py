_base_ = [
    './backbone_cfg.py',
]

# backbone
backbone_type = 'resnet'
backbone_cfg = _base_.backbone_cfgdict[backbone_type]

# neck
neck_type = None

# task head
taskhead_type = 'det'   # cls or det
taskhead_model = 'detr'
num_instance_queries = 50
eval_prime_score = 'coco/bbox_mAP'    # coco/bbox_mAP, coco/mAP
