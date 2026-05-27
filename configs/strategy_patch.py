
# strategy
lr = 5e-5
weight_decay = 0.001
max_epochs = 15
save_each_epoch = True
val_interval = 1

# loss_cfg = dict(
#     type='BCEWithLogitsLoss',
#     reduction='mean',
# )

loss_cfg = dict(
    type='AsymmetricLossOptimized',
    gamma_neg=2,
    gamma_pos=0,
    clip=0.0,
    eps=1e-5,
    disable_torch_grad_focal_loss=True,
    reduction='mean',
    loss_scale=1.0,
)


optim_wrapper = dict(    
    optimizer=dict(type='AdamW', lr=lr, weight_decay=weight_decay),
    clip_grad=dict(max_norm=1.0, norm_type=2),
    # paramwise_cfg=dict(
    #     custom_keys={'backbone': dict(lr_mult=10., decay_mult=1.0)})
)
# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=8*32)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        begin=0,
        end=1,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=max_epochs - 1,
        eta_min=1e-6,
        begin=1,
        end=max_epochs,
        by_epoch=True),
]

logger_name = 'wscer_patch'
load_from = None
