import os
import random
import glob

root_dir = '/home/zly/codes/cervix_wsi_cls/data_resource/cls_valid'
invalid_list = glob.glob(f'{root_dir}/invalid/*.png')
valid_list = glob.glob(f'{root_dir}/valid/*.png')
anno_list = glob.glob('/home/zly/codes/cervix_wsi_cls/data_resource/cls_pn/cut_img/random_cut/**/*.png')
# random_valid = random.sample(anno_list, 3000)
total_valid_list = [*valid_list, *anno_list]

random.shuffle(invalid_list)
random.shuffle(total_valid_list)

train_txt,val_txt = [],[]
for clsid,list_data in enumerate([invalid_list,total_valid_list]):
    train_data_num = int(0.9*len(list_data))
    for cnt,file_path in enumerate(list_data):

        if cnt < train_data_num:
            train_txt.append(f'{file_path} {clsid} \n')
        else:
            val_txt.append(f'{file_path} {clsid} \n')

random.shuffle(train_txt)
random.shuffle(val_txt)        
with open(f'{root_dir}/train.txt', 'w') as txtf:
    txtf.writelines(train_txt)
with open(f'{root_dir}/val.txt', 'w') as txtf:
    txtf.writelines(val_txt)