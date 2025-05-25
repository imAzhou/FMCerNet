import os
from PIL import Image
import json
import pandas as pd
from collections import defaultdict,Counter
import pickle
import numpy as np
from pycocotools import mask as mask_utils
from tqdm import tqdm
import cv2
import shutil
import random
import glob

WINDOW_SIZE = 750
POSITIVE_CLASS = ['AGC', 'ASC-US','LSIL', 'ASC-H', 'HSIL']
CLASS_COLORS = [[31,119,180], [255,153,153], [255,105,180], [255,20,147], [139,0,139]]

def coco_format(patchlist):
    format_result = {
        'categories': [{
            'id': idx+1,
            'name': clsname,
            'color': clscolor,
        } for idx, clsname,clscolor in zip(range(len(POSITIVE_CLASS)), POSITIVE_CLASS, CLASS_COLORS)],
        'images': [],
        'annotations': []
    }
    rle_output = {}
    annid = 0
    for idx,pInfo in enumerate(tqdm(patchlist, ncols=80)):
        format_result['images'].append(
            {'id': idx, 'file_name': pInfo['filename'], 'width': WINDOW_SIZE, 'height': WINDOW_SIZE,
             'prefix': pInfo['prefix'], 'diagnose': pInfo['diagnose'], 'maskfile': pInfo['maskfile']})
        sx1,sy1,sx2,sy2 = pInfo['square_coords']
        sw,sh = sx2-sx1, sy2-sy1
        # if sw!=WINDOW_SIZE or sh!=WINDOW_SIZE:
        #     print()
        if pInfo['diagnose'] == 1:
            inst_mask = np.load(f"data_resource/0511/WINDOW_SIZE_{WINDOW_SIZE}/patch_inst_mask/{pInfo['maskfile']}")['patch_mask']
            h,w = inst_mask.shape
            if h!=WINDOW_SIZE or w!=WINDOW_SIZE:
                print(f'Error size: (w,h): ({w},{h})')
                inst_mask = cv2.resize(inst_mask, (WINDOW_SIZE, WINDOW_SIZE), interpolation=cv2.INTER_NEAREST)
            rle_instdict = {}
            for inst_id in range(len(pInfo['bboxes'])):
                annmask = inst_mask == inst_id+1
                rle = mask_utils.encode(np.asfortranarray(annmask))
                rle['counts'] = rle['counts'].decode('utf-8')
                rle_instdict[inst_id+1] = rle
            rle_output[idx] = rle_instdict

        for bbox,clsname,inst_id in zip(pInfo['bboxes'],pInfo['clsnames'],range(len(pInfo['bboxes']))):
            x1,y1,x2,y2 = bbox
            w,h = x2-x1, y2-y1
            format_result['annotations'].append({
                "id": annid,
                "image_id": idx,
                'inst_id': inst_id+1,
                "category_id": POSITIVE_CLASS.index(clsname) + 1,
                "bbox": [x1,y1,w,h],
                "area": w*h,
                "iscrowd": 0,
            })
            annid += 1

    return format_result,rle_output


def main():
    # with open('data_resource/0511/WINDOW_SIZE_750/ann_jsons/patches_in_NegSlide.json', 'r', encoding='utf-8') as f:
    #     negslide_patchlist = json.load(f)
    with open('data_resource/0511/WINDOW_SIZE_750/ann_jsons/patches_in_RoI_pure_valid.json', 'r', encoding='utf-8') as f:
        RoI_patchlist = json.load(f)
    
    RoI_patchlist = filter_slide_neg(RoI_patchlist, neg_patch_thr=300) # 控制每张病人切片的阴性 patch 数量

    patient2patchlist = defaultdict(list)
    # for patchInfo in [*negslide_patchlist, *RoI_patchlist]:
    for patchInfo in RoI_patchlist:
        patient2patchlist[patchInfo['patientId']].append(patchInfo)
    
    data_group = {
        'puretrain': 'data_resource/0511/4_pure_train.csv',
        # 'fusiontrain': 'data_resource/0511/5_fusion_train.csv',
        'val': 'data_resource/0511/6_val.csv'
    }
    for tag,csvpath in data_group.items():
        df_data = pd.read_csv(csvpath)
        patchlist = []
        for row in tqdm(df_data.itertuples(index=False), total=len(df_data), ncols=80):
            patchlist.extend(patient2patchlist[row.patientId])
        patchInCOCO,rle_output = coco_format(patchlist)
        
        with open(f'data_resource/0511/WINDOW_SIZE_{WINDOW_SIZE}/annofiles/{tag}_coco.json', 'w', encoding='utf-8') as f:
            json.dump(patchInCOCO, f, ensure_ascii=False)
        with open(f'data_resource/0511/WINDOW_SIZE_{WINDOW_SIZE}/annofiles/{tag}_rle_masks.pkl', 'wb') as f:
            pickle.dump(rle_output, f)

def filter_slide_neg(RoI_patchlist, neg_patch_thr = 300):

    neg_count = Counter()
    for item in tqdm(RoI_patchlist, ncols=80):
        if item["prefix"] == 'neg':
            neg_count[item['patientId']] += 1
    filter_pids = [k for k, v in neg_count.items() if v > neg_patch_thr]
    random.shuffle(RoI_patchlist)
    
    new_RoI_patchlist = []
    filter_neg_count = Counter()
    for item in tqdm(RoI_patchlist, ncols=80):
        if item["prefix"] != 'neg':
            new_RoI_patchlist.append(item)
            continue

        pid = item['patientId']
        if pid in filter_pids and filter_neg_count[pid] >= neg_patch_thr:
            continue
        
        new_RoI_patchlist.append(item)
        filter_neg_count[pid] += 1
    
    return new_RoI_patchlist
    

def statistic():
    with open('data_resource/0511/WINDOW_SIZE_750/ann_jsons/patches_in_RoI_pure_valid.json', 'r', encoding='utf-8') as f:
        RoI_patchlist = json.load(f)
    pn_cnt = [0,0]
    RoI_patchlist = filter_slide_neg(RoI_patchlist, neg_patch_thr=300)
    for pInfo in RoI_patchlist:
        pn_cnt[pInfo['diagnose']] += 1
    print(pn_cnt)


def clear_imgs():
    keep_filename = []
    for tag in ['puretrain','val']:
        pn_cnt = [0,0]
        with open(f'{ann_dir}/{tag}_coco.json', 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        for patchinfo in tqdm(json_data['images'], ncols=80):
            keep_filename.append(patchinfo["file_name"])
            pn_cnt[patchinfo['diagnose']] += 1
        print(f'{tag}: {len(json_data["images"])}, [neg, pos]: [{pn_cnt[0]}, {pn_cnt[1]}]')
    
    exists_imgpath = glob.glob('data_resource/0511/WINDOW_SIZE_750/images/**/*.png')
    print(f'keep_filename nums: {len(keep_filename)}')
    print(f'exists_imgpath nums: {len(exists_imgpath)}')

    # for imgpath in tqdm(exists_imgpath, ncols=80):
    #     filename = os.path.basename(imgpath)
    #     if filename not in keep_filename:
    #         os.remove(imgpath)

if __name__ == "__main__":
    ann_dir = f'data_resource/0511/WINDOW_SIZE_{WINDOW_SIZE}/annofiles'
    os.makedirs(ann_dir, exist_ok=True, mode=0o777)
    
    # main()
    # statistic()
    clear_imgs()
    

'''
512:
puretrain: [98154, 38570]
fusiontrain: [142150, 103616]
val: [38189, 9919]

750:
puretrain: 66051, [neg, pos]: [30053, 35998]
val: 18536, [neg, pos]: [14630, 3906]
'''