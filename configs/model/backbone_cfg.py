_base_ = [
    './common_backbone_cfg.py',
    './histopathology_backbone_cfg.py',
    './cytopathology_backbone_cfg.py',
]

backbone_cfgdict = dict(
    **_base_.common_backbone_cfgdict,
    **_base_.histopathology_backbone_cfgdict,
    **_base_.cytopathology_backbone_cfgdict,
)
