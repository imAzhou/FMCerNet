# dataset settings 

# data_root = 'data_resource/0511/WINDOW_SIZE_750'
data_root = '/c22073/zly/datasets/CervicalDatasets/WINDOW_SIZE_1000'
img_dir = f'{data_root}/images'
classes = ['NILM', 'AGC', 'ASC-US', 'LSIL', 'ASC-H', 'HSIL']
num_classes = len(classes)
dataset_type = 'cls'    # cls, instance
train_bs = 32
val_bs = 32
input_size = 518  # 224, 392, 448, 512, 1024

train_annojson = f'{data_root}/annofiles/puretrain_cocoformat.json'
rand_increasing_policies = [
    dict(type='AutoContrast'),
    dict(type='Equalize'),
    dict(type='Rotate', magnitude_key='angle', magnitude_range=(0, 30), prob=0.5),
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

train_transform = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(input_size, input_size), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandAugment',
        policies=rand_increasing_policies,
        num_policies=2,
        magnitude_level=5),
    dict(type='PackInputs')
]

val_annojson = f'{data_root}/annofiles/val_cocoformat.json'
val_transform = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(input_size, input_size), keep_ratio=True),
    dict(type='PackInputs')
]

test_transform = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(input_size, input_size), keep_ratio=True),
    dict(type='PackInputs')
]
