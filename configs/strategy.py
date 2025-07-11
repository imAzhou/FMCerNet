
# strategy
lr = 0.0001
weight_decay = 0.0001
max_epochs = 50
save_each_epoch = False
val_interval = 5

optim_wrapper = dict(    
    optimizer=dict(type='AdamW', lr=lr, weight_decay=weight_decay),
    clip_grad=dict(max_norm=1.0, norm_type=2),  # 添加梯度裁剪
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)})
)
# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=8*16)

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[100],
        gamma=0.1)
]

logger_name = 'wscer_partial'
apply_auxiliary = 'random'  # random, logit
load_from = 'checkpoints/detr_r50_rename.pth'
