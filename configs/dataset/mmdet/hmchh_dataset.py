# dataset settings 

data_root = 'data_resource/HMCHH'
img_dir = f'{data_root}/JPEGImages'
classes = ['abnormal',]
num_classes = len(classes)
dataset_type = 'instance'    # cls, instance
train_bs = 8
val_bs = 8
with_mask = False
input_size = 1024  # 224, 392, 448, 512, 1024

train_annojson = f'{data_root}/annofiles_roi/fold1_train.json'
albu_train_transforms = [
    dict(type='ShiftScaleRotate',
        shift_limit=0.0625,
        scale_limit=0.0,
        rotate_limit=15,    # 随机在 [-15°, +15°] 之间旋转；
        interpolation=1,
        p=0.5),
    dict(type='RandomBrightnessContrast',
        brightness_limit=[0.8, 1.5],
        # contrast_limit=[-0.5, 0.5],
        p=0.2),
    # dict(type='OneOf',
    #     transforms=[
    #         dict(
    #             type='RGBShift',
    #             r_shift_limit=10,
    #             g_shift_limit=10,
    #             b_shift_limit=10,
    #             p=1.0),
    #         dict(
    #             type='HueSaturationValue',
    #             hue_shift_limit=20,
    #             sat_shift_limit=30,
    #             val_shift_limit=20,
    #             p=1.0)
    #     ],
    #     p=0.1),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=5, p=1.0),
            dict(type='MedianBlur', blur_limit=5, p=1.0)
        ],
        p=0.5),
]

train_transform = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=with_mask),
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

val_annojson = f'{data_root}/annofiles_roi/fold1_val.json'
val_transform = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=with_mask),
    dict(type='Resize', scale=(input_size, input_size), keep_ratio=True),
    dict(type='PackDetInputs')
]

test_transform = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=with_mask),
    dict(type='Resize', scale=(input_size, input_size), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

# val_annojson_roi = f'{data_root}/annofiles_roi/fold1_val_cocoformat.json'
val_evaluator = dict(
    ann_file=val_annojson,
    metric='bbox',
    classwise=False,
    metric_items = ['mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l', 'AR@1000'],
    format_only=False,)