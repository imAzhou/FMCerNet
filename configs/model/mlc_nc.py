_base_ = [
    './backbone_cfg.py',
]

net_type = 'patch'
# backbone
backbone_type = 'smartccs'
backbone_cfg = _base_.backbone_cfgdict[backbone_type]
backbone_cfg['frozen_backbone'] = True
backbone_cfg['use_peft'] = None

# classifier
taskhead_model = 'mlc_nc'
positive_thr = 0.5
eval_prime_score = 'multi-label/f1-score'

mlc_nc_cfg = dict(
    classifier='ETF',
    with_background=True,
    embed_dim=768,
    project_dim=20,
    num_heads=8,
    dropout=0.1,
    alpha=1.0,
    weight_twoway=1.0,
    weight_fla=0.5,
    weight_prototype=0.1,
    prototype_momentum=0.9,
    prototype_temperature=0.5,
    two_way_tp=4.0,
    two_way_tn=1.0,
)
