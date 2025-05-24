# dataset settings 

data_root = 'data_resource/0511/WINDOW_SIZE_750'
# data_root = '/c22073/zly/datasets/CervicalDatasets/LCerScanv1_512'
img_dir = f'{data_root}/images'
instance_mask_dir = f'{data_root}/patch_inst_mask'
classes = ['NILM', 'AGC', 'ASC-US', 'LSIL', 'ASC-H', 'HSIL']
num_classes = len(classes)
dataset_type = 'instance'    # cls, instance
train_bs = 16
val_bs = 16
input_size = 750  # 224, 392, 448, 512, 1024

train_annojson = f'{data_root}/annofiles/puretrain_coco.json'
train_rel_file = f'{data_root}/annofiles/puretrain_rle_masks.pkl'
train_transform = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    # dict(type='Resize', scale=(input_size, input_size), keep_ratio=True),
    dict(type='Resize', scale=(input_size, input_size)),
    # dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

val_annojson = f'{data_root}/annofiles/val_coco.json'
val_rel_file = f'{data_root}/annofiles/val_rle_masks.pkl'
val_transform = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(input_size, input_size)),
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