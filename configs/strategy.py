
# strategy
lr = 0.0001
min_lr = 0.00001
weight_decay = 0.001
max_epochs = 100
warmup_epoch = 5
gamma = 0.95
save_each_epoch = False
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=lr, weight_decay=weight_decay),
    # clip_grad=dict(max_norm=1.0, norm_type=2)  # 添加梯度裁剪
)

positive_thr = 0.5
<<<<<<< HEAD
img_size = 224  # 224, 448, 512, 1024
=======
>>>>>>> 8de63f31a2f377a4e101bc5e333242bb1a61d3bf

logger_name = 'wscer_partial'
apply_auxiliary = 'random'  # random, logit
load_from = None
# eval_prime_score = 'single-label/binary_accuracy'
# multi-label/img_accuracy
eval_prime_score = 'multi-label/img_accuracy'