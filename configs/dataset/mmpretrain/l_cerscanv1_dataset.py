# dataset settings 

data_root = 'data_resource/WINDOW_SIZE_1600'
img_dir = f'{data_root}/images'
classes = ['AGC', 'ASC-US', 'LSIL', 'ASC-H', 'HSIL']
num_classes = len(classes)
dataset_type = 'multicls'    # cls, instance
train_bs = 32
val_bs = 32
input_size = 448  # 224, 392, 448, 512, 1024
# train_annfile = 'hardsample_annofiles/multilable_hs_round1.json'
# train_annfile = 'annofiles/multilabel_puretrain.json'
train_annfile = 'hardsample_annofiles/hs_round1_hicervix_otsu.json'

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
    ann_file = train_annfile,
    pipeline = [
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
    data_root = data_root,
    data_prefix = 'images',
    ann_file = 'annofiles/multilabel_val.json',
    pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='Resize', scale=(input_size, input_size), keep_ratio=True),
        dict(type='PackInputs')
    ]
)
