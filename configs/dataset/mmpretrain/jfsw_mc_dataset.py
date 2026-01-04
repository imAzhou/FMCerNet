# dataset settings 
data_root = 'data_resource/cell_attri/cell_inst'
classes = ['NILM', 'GEC', 'AGC', 'ASC-US', 'LSIL', 'ASC-H', 'HSIL']
num_classes = len(classes)
dataset_type = 'cls'    # cls, instance
train_bs = 64
val_bs = 64
input_size = 224  # 224, 392, 448, 512, 1024

rand_increasing_policies = [
    dict(type='AutoContrast', prob=0.5),
    dict(type='Equalize', prob=0.5),
    dict(type='Rotate', magnitude_key='angle', magnitude_range=(-15, 15), prob=0.5),
    dict(type='Contrast', magnitude_range=(-0.9, 0.9), prob=0.5),
    dict(type='Brightness', magnitude_key='magnitude', magnitude_range=(-0.9, 0.9), prob=0.5),
    dict(type='Sharpness', magnitude_key='magnitude', magnitude_range=(0, 9), prob=0.5),
    dict(
        type='Shear',
        magnitude_key='magnitude',
        magnitude_range=(-0.9, 0.9),
        direction='horizontal', prob=0.5),
    dict(
        type='Shear',
        magnitude_key='magnitude',
        magnitude_range=(-0.9, 0.9),
        direction='vertical', prob=0.5),
]

train_datasets = dict(
    data_root = data_root,
    data_prefix = 'images',
    ann_file = 'filter_train.txt',
    classes = classes,
    pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='ResizeEdge', edge='long', scale=input_size),
        dict(type='CenterCrop', auto_pad=True, crop_size=input_size, pad_cfg=dict(pad_val=255, type='Pad')),
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
    data_root = data_root,
    data_prefix = 'images',
    ann_file = 'filter_val.txt',
    classes = classes,
    pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='ResizeEdge', edge='long', scale=input_size),
        dict(type='CenterCrop', auto_pad=True, crop_size=input_size, pad_cfg=dict(pad_val=255, type='Pad')),
        dict(type='PackInputs')
    ]
)
