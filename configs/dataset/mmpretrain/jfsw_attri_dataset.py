# dataset settings 
data_root = 'data_resource/cell_attri/cell_inst'
classes = ['NILM', 'GEC', 'AGC', 'ASC-US', 'LSIL', 'ASC-H', 'HSIL']
num_classes = len(classes)
img_dir = f'{data_root}/images'
dataset_type = 'attricls'    # cls, instance
attribute_names = ["Nsize","Nstains","Nchromatin","Nregular","cytoplasm","arrangement","polarity","cellType"]
attribute_classes = [5,3,3,2,5,4,2,7]
custom_weights = [[1.0 for i in range(num)] for num in attribute_classes]
# Attr 4: 细胞浆状态
custom_weights[4] = [1.0, 10.0, 10.0, 10.0, 10.0]
num_attributes = len(attribute_classes)
train_bs = 64
val_bs = 64
input_size = 224

train_annfile = f'{data_root}/train_cellinst.json'
val_annfile = f'{data_root}/val_cellinst.json'

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

train_transform = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(input_size, input_size)),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandAugment',
        policies=rand_increasing_policies,
        num_policies=2,
        magnitude_level=5),
    dict(type='PackInputs')
]

val_transform = [
        dict(type='LoadImageFromFile'),
        dict(type='Resize', scale=(input_size, input_size)),
        dict(type='PackInputs')
]
