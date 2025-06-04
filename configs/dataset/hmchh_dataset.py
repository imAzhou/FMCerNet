# dataset settings 

data_root = 'data_resource/HMCHH/WINDOW_SIZE_512'
# data_root = '/c22073/zly/datasets/CervicalDatasets/LCerScanv1_750'
img_dir = f'{data_root}/images'
classes = ['normal', 'abnormal']
num_classes = len(classes)
dataset_type = 'instance'    # cls, instance
train_bs = 16
val_bs = 16
input_size = 512  # 224, 392, 448, 512, 1024

train_annojson = f'{data_root}/annofiles_tile/fold1_train_cocoformat.json'
train_transform = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(input_size, input_size), keep_ratio=True),
    # dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

val_annojson = f'{data_root}/annofiles_tile/fold1_val_cocoformat.json'
val_transform = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(input_size, input_size), keep_ratio=True),
    dict(type='PackDetInputs')
]

test_transform = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(input_size, input_size), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

val_annojson_roi = f'{data_root}/annofiles_roi/fold1_val_cocoformat.json'
val_evaluator = dict(
    ann_file=val_annojson_roi,
    metric='bbox',
    classwise=False,
    metric_items = ['mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l', 'AR@1000'],
    format_only=False,)