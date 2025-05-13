# dataset settings 

data_root = 'data_resource/0429/512'
# data_root = '/c22073/zly/datasets/CervicalDatasets/LCerScanv1_512'
classes = ['NILM', 'AGC', 'ASC-US', 'LSIL', 'ASC-H', 'HSIL']
num_classes = len(classes)
train_bs = 16
val_bs = 16
split_group = 1     # 一般情况下设置为1，不等于1时，会分 split_group 组数据分别依次送入backbone抽特征后，组合到一起送入解码器中训练

train_annojson = 'train.json'
val_annojson = 'val.json'

dataset_type = 'instance'    # cls, instance