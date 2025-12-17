data_root = 'data_resource/0630/WINDOW_SIZE_1600'
feat_dir = f'{data_root}/slide_feat_ours'
classes = ['NILM', 'AGC', 'ASC-US', 'LSIL', 'ASC-H', 'HSIL']
cls_map = {
    'NILM': 'NILM',
    'ASC-US': 'ASC-US',
    'LSIL': 'LSIL',
    'ASC-H': 'ASC-H',
    'HSIL': 'HSIL',
    'SCC': 'HSIL',
    'AGC-N': 'AGC',
    'AGC': 'AGC',
    'AGC-NOS': 'AGC',
    'AGC-FN': 'AGC',
}
num_classes = len(classes)
dataset_type = 'slide'    # cls, instance
patch_nums = 1000
train_bs = 32
val_bs = 32

train_csvfile = 'data_resource/0630/45_0924_train.csv'
val_csvfile = 'data_resource/0630/67_0924_val.csv'
