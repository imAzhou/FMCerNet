data_root = 'data_resource/WINDOW_SIZE_850'
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
patch_nums = 1000
train_bs = 32
val_bs = 32

train_csvfile = f'{data_root}/annofiles/45_purejfsw_train.csv'
val_csvfile = f'{data_root}/annofiles/67_wsi_val.csv'
