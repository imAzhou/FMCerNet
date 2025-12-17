
# strategy
lr = 0.0001
weight_decay = 0.00001
max_epochs = 100
save_each_epoch = False
val_interval = 5

optim_wrapper = dict(    
    optimizer=dict(type='AdamW', lr=lr, weight_decay=weight_decay),
)

auto_scale_lr = None

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[25, 45],
        gamma=0.1)
]

logger_name = 'wscer_partial'
# apply_auxiliary = 'random'  # random, logit
# load_from = 'checkpoints/detr_r50_rename.pth'
load_from = None
eval_prime_score = 'f1_macro'
format_type = 'pn_pos'  # direct,pn_pos,nopn_pos
