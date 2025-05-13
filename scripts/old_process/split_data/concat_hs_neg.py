import os
import glob
from tqdm import tqdm
import shutil
import random
import cv2
import albumentations as A


def cat_anno_txt():
    root_dir = 'data_resource/cls_pn/cut_img'

    with open(f'{root_dir}/train_rcp_c6.txt', 'r') as f:
        exist_lines = f.readlines()
    cat_lines = [*exist_lines, *hs_train_list]
    random.shuffle(cat_lines)

    with open(f'{root_dir}/train_rcp_hs_c6.txt', 'w') as f:
        f.writelines(cat_lines)

def gene_hs_list():
    hard_sampls_dir = 'predict_results/w500'
    rc_target_dir = 'data_resource/cls_pn/cut_img/random_cut/hs_NILM'
    os.makedirs(rc_target_dir, exist_ok=True)
    ori_target_dir = 'data_resource/cls_pn/cut_img/original/hs_NILM'
    os.makedirs(ori_target_dir, exist_ok=True)

    hs_train_list = []
    for patientId in tqdm(os.listdir(hard_sampls_dir), ncols=80):
        hs_imgs = glob.glob(f'{hard_sampls_dir}/{patientId}/1/**.png')
        for idx,img_path in enumerate(hs_imgs):
            img_name = f'{patientId}_hs{idx}.png'
            image = cv2.imread(img_path)

            min_len,max_len = 300,500
            new_w,new_h = random.randint(min_len,max_len),random.randint(min_len,max_len)
            random_crop = A.RandomCrop(width=new_w, height=new_h)
            cropped_image = random_crop.apply(image)
            cv2.imwrite(f'{rc_target_dir}/{img_name}', cropped_image)

            shutil.copy(img_path, f'{ori_target_dir}/{img_name}')
            hs_train_list.append(f'hs_NILM/{img_name} 0\n')

    return hs_train_list

if __name__ == '__main__':
    hs_train_list = gene_hs_list()
    cat_anno_txt()
