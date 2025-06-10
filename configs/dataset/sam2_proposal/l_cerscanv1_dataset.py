# dataset settings 

# data_root = 'data_resource/0511/WINDOW_SIZE_1000'
# data_root = '/c22073/zly/datasets/CervicalDatasets/WINDOW_SIZE_1000'
data_root = '/c23030/zly/datasets/CervicalDatasets/WINDOW_SIZE_1000'
img_dir = f'{data_root}/images'
classes = ['NILM', 'AGC', 'ASC-US', 'LSIL', 'ASC-H', 'HSIL']
num_classes = len(classes)
dataset_type = 'instance'    # cls, instance
load_proposal = True
train_bs = 4
val_bs = 4
input_size = 1024  # 224, 392, 448, 512, 1024

train_annojson = f'{data_root}/annofiles/fusiontrain_cocoformat.json'

train_transform = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(input_size, input_size), keep_ratio=True),
    dict(type='PackDetInputs')
]

val_annojson = f'{data_root}/annofiles/val_cocoformat.json'
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

val_evaluator = dict(
    ann_file=val_annojson,
    # metric='bbox',
    metric='proposal',
    classwise=False,
    iou_thrs=[0.3],
    # metric_items = ['mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l', 'AR@1000'],
    format_only=False,)