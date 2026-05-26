# dataset settings
data_root = 'data_resource/LCerScan/WS1600'
classes = ['AGC', 'ASC-US', 'LSIL', 'ASC-H', 'HSIL']
num_classes = len(classes)
dataset_type = 'multicls'    # cls, instance
train_bs = 32
val_bs = 32
input_size = 1024  # 224, 392, 448, 512, 1024

rand_increasing_policies = [
    dict(type='AutoContrast', prob=0.3),
    dict(type='Rotate', magnitude_range=(0, 15), prob=0.5, interpolation='bicubic'),
    dict(type='Contrast', magnitude_range=(0, 0.3), prob=0.5),
    dict(type='Brightness', magnitude_range=(0, 0.3), prob=0.5),
    dict(type='Sharpness', magnitude_range=(0, 0.6), prob=0.5),
    dict(type='ColorTransform', magnitude_range=(0, 0.2), prob=0.5),
    dict(
        type='Shear',
        magnitude_range=(0, 0.15),
        direction='horizontal', prob=0.5),
    dict(
        type='Shear',
        magnitude_range=(0, 0.15),
        direction='vertical', prob=0.5),
]

train_datasets = dict(
    data_root=data_root,
    data_prefix='images',
    ann_file='annofiles/multilabel_puretrain.json',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='Resize', scale=(input_size, input_size), keep_ratio=True),
        dict(type='RandomFlip', prob=0.5),
        dict(
            type='RandAugment',
            policies=rand_increasing_policies,
            num_policies=2,
            magnitude_level=5),
        dict(type='PackInputs')
    ],
)

val_datasets = dict(
    data_root=data_root,
    data_prefix='images',
    ann_file='annofiles/multilabel_val.json',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='Resize', scale=(input_size, input_size), keep_ratio=True),
        dict(type='PackInputs')
    ]
)
