import os
import time
import pandas as pd
import torch
from cerwsi.nets import ValidClsNet
from cerwsi.utils import KFBSlide
from PIL import Image
import random
import cv2
import numpy as np
from mmpretrain.structures import DataSample
import json
import warnings
from tqdm import tqdm

LEVEL = 0
CERTAIN_THR = 0.7
PATCH_EDGE = 750
SAFE_MARGIN = 100
cut_nums_each = 40
img_save_dir = f'data_resource/0511/WINDOW_SIZE_750/images/neg_slide'
os.makedirs(img_save_dir, exist_ok=True, mode=0o777)
anno_save_dir = 'data_resource/0511/WINDOW_SIZE_750/ann_jsons'
os.makedirs(anno_save_dir, exist_ok=True, mode=0o777)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.conv")

def cut_random_neg():
    device = torch.device('cuda:0')
    valid_model_ckpt = 'checkpoints/valid_cls_best.pth'
    valid_model = ValidClsNet()
    valid_model.to(device)
    valid_model.eval()
    valid_model.load_state_dict(torch.load(valid_model_ckpt))

    train_csv = pd.read_csv('data_resource/0511/4_pure_train.csv')
    val_csv = pd.read_csv('data_resource/0511/6_val.csv')
    filtered = {
        'train': train_csv[train_csv['kfb_clsid'] == 0],
        'val': val_csv[val_csv['kfb_clsid'] == 0],
    }

    total_patches = 0
    neg_patch_list = []
    txt_records = []
    low_valid_records = []
    
    for mode in ['train','val']:
        filtered[mode] = filtered[mode].reset_index(drop=True)
        for r_idx, row in filtered[mode].iterrows():
            if r_idx > 1:
                break
            slide_patch_cnt = 0
            kfb_path, patientId = row['kfb_path'], row['patientId']
            slide = KFBSlide(kfb_path)
            max_x, max_y = slide.level_dimensions[LEVEL]
            max_x, max_y = max_x-SAFE_MARGIN, max_y-SAFE_MARGIN
            start_time = time.time()
            for i in tqdm(range(cut_nums_each), ncols=80):
                x1,y1 = random.randint(SAFE_MARGIN, max_x-PATCH_EDGE),random.randint(SAFE_MARGIN, max_y-PATCH_EDGE)
                read_result = Image.fromarray(slide.read_region((x1,y1), LEVEL, (PATCH_EDGE,PATCH_EDGE)))
                data_batch = dict(inputs=[], data_samples=[])
                img_input = cv2.cvtColor(np.array(read_result), cv2.COLOR_RGB2BGR)
                img_input = torch.as_tensor(cv2.resize(img_input, (224,224)))
                data_batch['inputs'].append(img_input.permute(2,0,1))    # (bs, 3, h, w)
                data_batch['data_samples'].append(DataSample())
                data_batch['inputs'] = torch.stack(data_batch['inputs'], dim=0)
                with torch.no_grad():
                    outputs = valid_model.val_step(data_batch)
            
                if max(outputs[0].pred_score) > CERTAIN_THR and outputs[0].pred_label == 1:
                    filename = f'{patientId}_{slide_patch_cnt}.png'
                    neg_patch_list.append({
                        'patientId': patientId,
                        'media_type': 'slide',
                        'source_path': kfb_path,
                        'filename': filename,
                        'square_coords': (x1,y1,x1+PATCH_EDGE,y1+PATCH_EDGE),    # 在媒体资源中的相对坐标
                        'bboxes': [],
                        'clsnames': [],
                        'prefix': 'neg',
                        'diagnose': 0,
                        'maskfile': ''
                    })
                    read_result.save(f'{img_save_dir}/{filename}')
                    slide_patch_cnt += 1

            t_delta = time.time() - start_time
            print(f'[{r_idx+1}/{len(filtered[mode])}] {patientId} cut nums: {slide_patch_cnt}, cost: {t_delta:0.2f}s')
            txt_records.append(f'{patientId} cut nums: {slide_patch_cnt}. \n')
            if slide_patch_cnt < 5:
                low_valid_records.append(f'{patientId} cut nums: {slide_patch_cnt}. \n')
            total_patches += slide_patch_cnt
        
    print(f'Total: {total_patches} patches. \n')
    txt_records.append(f'Total: {total_patches} patches. \n')
    with open(f'{anno_save_dir}/neg_patch_nums.txt', 'w') as f:
        f.writelines(txt_records)
        f.writelines(f"{'='*20} \n\n\n")
        f.writelines(low_valid_records)

    with open(f'{anno_save_dir}/patches_in_NegSlide.json', 'w') as f:
        json.dump(neg_patch_list, f)

# def retrieve_anojson():
#     all_negs = os.listdir('data_resource/0511/WINDOW_SIZE_750/images/neg')
#     with open('data_resource/0511/WINDOW_SIZE_750/ann_jsons/patches_in_RoI_pure_validjson', 'r', encoding='utf-8') as f:
#         json_data = json.load(f)
#     for pInfo in tqdm(json_data, ncols=80):
#         print()


if __name__ == '__main__':
    cut_random_neg()

