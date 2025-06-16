# dataset settings 

data_root = 'data_resource/0511/WINDOW_SIZE_512'
# data_root = '/c22073/zly/datasets/CervicalDatasets/WINDOW_SIZE_1000'
img_dir = f'{data_root}/images'
classes = ['NILM', 'AGC', 'ASC-US', 'LSIL', 'ASC-H', 'HSIL']
num_classes = len(classes)
dataset_type = 'instance'    # cls, instance
load_proposal = False
train_bs = 128
val_bs = 128
input_size = 224  # 224, 392, 448, 512, 1024

train_annojson = f'{data_root}/annofiles/fusiontrain_cocoformat.json'
albu_train_transforms = [
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.0625,
        scale_limit=0.0,
        rotate_limit=15,    # 随机在 [-15°, +15°] 之间旋转；
        interpolation=1,
        p=0.5),
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.2),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='RGBShift',
                r_shift_limit=10,
                g_shift_limit=10,
                b_shift_limit=10,
                p=1.0),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0)
        ],
        p=0.1),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0)
        ],
        p=0.1),
]
train_transform = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(input_size, input_size), keep_ratio=True),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',    # bbox: [x1,y1,x2,y2]
            label_fields=['gt_bboxes_labels', 'gt_ignore_flags'],
            min_visibility=0.0),
        keymap={
            'img': 'image',
            'gt_masks': 'masks',
            'gt_bboxes': 'bboxes'
        },
        skip_img_without_anno=False),
    dict(type='RandomFlip', prob=0.5),
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