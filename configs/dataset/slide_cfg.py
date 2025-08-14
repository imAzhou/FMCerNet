data_root = 'data_resource/0630/WINDOW_SIZE_850'
feat_dir = f'{data_root}/slide_feat'
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
train_bs = 32
val_bs = 32

train_csvfile = f'{data_root}/annofiles/4_pure_train.csv'
val_csvfile = f'{data_root}/annofiles/6_val.csv'
