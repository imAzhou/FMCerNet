_base_ = [
    './backbone_cfg.py',
]

net_type = 'patch'
# backbone
backbone_type = 'fusionnet'
# backbone_type = 'smartccs'    # smartccs, cytofm, unicas
backbone_cfg = _base_.backbone_cfgdict[backbone_type]
backbone_cfg['frozen_backbone'] = True
backbone_cfg['use_peft'] = None     # 'lora', None

# classifier
taskhead_model = 'wscer_mlc'
key_gate_scale = 1.0
positive_thr = 0.5
format_heatmap = False
format_img_token = False
eval_prime_score = 'multi-label/f1-score'
