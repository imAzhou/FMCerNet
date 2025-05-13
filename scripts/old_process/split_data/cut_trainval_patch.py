import json
import os
from tqdm import tqdm
import pandas as pd
import torch
from cerwsi.nets import ValidClsNet
from prettytable import PrettyTable
from cerwsi.utils import KFBSlide, random_cut_fn
from PIL import Image
import random
import cv2
import numpy as np
from mmpretrain.structures import DataSample
import time

RANDOM_CUT_POSITIVE = True
CUT_NUM = 2
NEGATIVE_CLASS = ['NILM', 'GEC']
ASC_CLASS = ['ASC-US', 'LSIL', 'ASC-H', 'HSIL', 'SCC']
AGC_CLASS = ['AGC-NOS', 'AGC', 'AGC-N', 'AGC-FN']

def cut_patch(valid_imgs, mode):
    save_dir = 'data_resource/cls_pn/cut_img'
    original_save_path = f'{save_dir}/original'
    os.makedirs(original_save_path, exist_ok=True)
    if RANDOM_CUT_POSITIVE:
        random_cut_save_path = f'{save_dir}/random_cut'
        os.makedirs(random_cut_save_path, exist_ok=True)

    mode_txt_lines,mode_rcp_txt_lines = [],[]
    for slide_valid_item in tqdm(valid_imgs, ncols=80):
        patientId = slide_valid_item['patientId']
        slide = KFBSlide(slide_valid_item['kfb_path'])

        for idx, img_item in enumerate(slide_valid_item['valid_anno']):
            patch_clsname = img_item['patch_clsname']
            x1,y1,x2,y2 = img_item['coord']
            w,h = img_item['size']
            location, level, size = (x1,y1), 0, img_item['size']
            read_result = Image.fromarray(slide.read_region(location, level, size))
            filename = f'{patientId}_{idx}.png'
            patch_clsid = 1
            if patch_clsname in NEGATIVE_CLASS:
                filename = f'{patientId}_anno{idx}.png'
                patch_clsid = 0

            os.makedirs(f'{original_save_path}/{patch_clsname}', exist_ok=True)
            read_result.save(f'{original_save_path}/{patch_clsname}/{filename}')
            mode_txt_lines.append(f'{patch_clsname}/{filename} {patch_clsid}\n')

            if RANDOM_CUT_POSITIVE and patch_clsname not in NEGATIVE_CLASS:
                cut_results = random_cut_fn(int(x1),int(y1),int(w),int(h),CUT_NUM)
                for j,new_rect in enumerate(cut_results):
                    new_x1,new_y1,new_w,new_h = new_rect
                    location, level, size = (new_x1,new_y1), 0, (new_w,new_h)
                    read_result = Image.fromarray(slide.read_region(location, level, size))
                    filename = f'{patientId}_rc{idx}{j}.png'
                    os.makedirs(f'{random_cut_save_path}/{patch_clsname}', exist_ok=True)
                    read_result.save(f'{random_cut_save_path}/{patch_clsname}/{filename}')
                    mode_rcp_txt_lines.append(f'{patch_clsname}/{filename} 1\n')
    
    with open(f'{save_dir}/anno_{mode}.txt', 'w') as txtf:
        txtf.writelines(mode_txt_lines)
    with open(f'{save_dir}/rcp_anno_{mode}.txt', 'w') as txtf:
        txtf.writelines(mode_rcp_txt_lines)

def cut_anno_json():
    with open('data_resource/cls_pn/1127_anno_train.json','r') as f:
        train_data = json.load(f)
    with open('data_resource/cls_pn/1127_anno_val.json','r') as f:
        val_data = json.load(f)
    
    cut_patch(train_data['valid_imgs'], 'train')
    cut_patch(val_data['valid_imgs'], 'val')

def cut_random_neg():
    device = torch.device('cuda:0')
    valid_model_ckpt = 'checkpoints/vlaid_cls_best.pth'
    valid_model = ValidClsNet()
    valid_model.to(device)
    valid_model.eval()
    valid_model.load_state_dict(torch.load(valid_model_ckpt))
    CERTAIN_THR = 0.7

    root_dir = 'data_resource/cls_pn'
    img_save_dir = f'{root_dir}/cut_img/random_cut/rc_NILM'
    os.makedirs(img_save_dir, exist_ok=True)
    train_csv = pd.read_csv(f'{root_dir}/1127_train.csv')
    val_csv = pd.read_csv(f'{root_dir}/1127_val.csv')
    filtered = {
        'train': train_csv[train_csv['kfb_clsid'] == 0],
        'val': val_csv[val_csv['kfb_clsid'] == 0],
    }
    valid_idx = 0
    for mode in ['train','val']:
        mode_txt_lines = []
        total_nums = len(filtered[mode])
        filtered[mode] = filtered[mode].reset_index(drop=True)
        for r_idx, row in filtered[mode].iterrows():
            kfb_path, patientId = row['kfb_path'], row['patientId']
            slide = KFBSlide(kfb_path)
            max_x, max_y = slide.level_dimensions[0]
            # 100 张宽高在[100,300]， 200 张宽高在[300,600]
            small_cut_num, large_cut_num = 50, 140
            if mode == 'val':
                small_cut_num, large_cut_num = 30, 70
            for random_cut_num in [(small_cut_num, [100,300]), (large_cut_num, [300,600])]:
                for i in tqdm(range(random_cut_num[0]), ncols=80, desc=f'Slide Num({mode}) {r_idx + 1}/{total_nums}'):
                    x1,y1 = random.randint(0, max_x),random.randint(0, max_y)
                    w = random.randint(random_cut_num[1][0], random_cut_num[1][1])
                    h = int(w*random.uniform(0.8, 1.5))
                    if x1+w > max_x or y1+h > max_y:
                        continue
                    read_result = Image.fromarray(slide.read_region((x1,y1), 0, (w,h)))

                    img_input = cv2.cvtColor(np.array(read_result), cv2.COLOR_RGB2BGR)
                    img_input = torch.as_tensor(cv2.resize(img_input, (224,224)))
                    data_batch = dict(inputs=[], data_samples=[])
                    data_batch['inputs'].append(img_input.permute(2,0,1))    # (bs, 3, h, w)
                    data_batch['data_samples'].append(DataSample())
                    data_batch['inputs'] = torch.stack(data_batch['inputs'], dim=0)
                    with torch.no_grad():
                        outputs = valid_model.val_step(data_batch)
                    if max(outputs[0].pred_score) > CERTAIN_THR and outputs[0].pred_label == 1:
                        unique_tag = str(time.time()).split('.')[1]
                        filename = f'{patientId}_{unique_tag}{valid_idx}.png'
                        
                        # batch_dir = f'{img_save_dir}/{valid_idx//1000}'
                        # os.makedirs(batch_dir, exist_ok=True)
                        read_result.save(f'{img_save_dir}/{filename}')
                        mode_txt_lines.append(f'rc_NILM/{filename} 0\n')
                        valid_idx += 1
        
        with open(f'{root_dir}/cut_img/neg_rc_{mode}.txt', 'w') as txtf:
            txtf.writelines(mode_txt_lines)



if __name__ == '__main__':
    # cut_anno_json()
    cut_random_neg()