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
from prettytable import PrettyTable

<<<<<<< HEAD
WINDOW_SIZE = 1600
POSITIVE_CLASS = ['AGC', 'ASC-US','LSIL', 'ASC-H', 'HSIL']
CLASS_COLORS = [[31,119,180], [255,153,153], [255,105,180], [255,20,147], [139,0,139]]
data_root = f'data_resource/0630/WINDOW_SIZE_{WINDOW_SIZE}'
neg_patch_thr = 3
=======
WINDOW_SIZE = 850
data_root = f'data_resource/WINDOW_SIZE_{WINDOW_SIZE}'
POSITIVE_CLASS = ['AGC', 'ASC-US','LSIL', 'ASC-H', 'HSIL']
CLASS_COLORS = [[31,119,180], [255,153,153], [255,105,180], [255,20,147], [139,0,139]]
neg_patch_thr = 10


def filter_slide_neg(RoI_patchlist, neg_patch_thr = 300):
    if neg_patch_thr == 0:
        new_RoI_patchlist = [i for i in RoI_patchlist if i["prefix"] != 'neg']
        return new_RoI_patchlist

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
>>>>>>> e14f4888d1e4228c257149865d6deb152971c162

def main():
    with open(f'{data_root}/ann_jsons/patches_in_RoI_pure_valid.json', 'r', encoding='utf-8') as f:
        RoI_patchlist = json.load(f)
    with open(f'{data_root}/ann_jsons/patches_in_RoI_jfsw_valid.json', 'r', encoding='utf-8') as f:
        jfsw_pos_patchdata = json.load(f)
<<<<<<< HEAD
    
    patient2patchlist = defaultdict(list)
    for item in RoI_patchlist:
        patient2patchlist[item['patientId']].append(item)
=======

    # RoI_patchlist = filter_slide_neg(RoI_patchlist, neg_patch_thr=neg_patch_thr) # 控制每张病人切片的阴性 patch 数量

    patient2patchlist = defaultdict(list)
    for patchInfo in RoI_patchlist:
        patient2patchlist[patchInfo['patientId']].append(patchInfo)
>>>>>>> e14f4888d1e4228c257149865d6deb152971c162
    
    data_group = {
        'puretrain': 'data_resource/0630/4_pure_train.csv',
        'val': 'data_resource/0630/6_val.csv'
    }
<<<<<<< HEAD
    
    for tag,csvpath in data_group.items():
        multilabel_pn_cnt, binary_pn_cnt = [0,0],[0,0]
=======
    multilabel_pn_cnt, binary_pn_cnt = []
    for tag,csvpath in data_group.items():
>>>>>>> e14f4888d1e4228c257149865d6deb152971c162
        df_data = pd.read_csv(csvpath)
        print(f'Load {tag} patchlist...')
        patchlist = []
        for row in tqdm(df_data.itertuples(index=False), total=len(df_data), ncols=80):
<<<<<<< HEAD
            samplelist = patient2patchlist[row.patientId]
            if tag == 'puretrain' and neg_patch_thr > 0:
                poslist = [i for i in samplelist if i['diagnose']==1]
                neglist = [i for i in samplelist if i['diagnose']==0]
                random.shuffle(neglist)
                samplelist = [*poslist, *neglist[:neg_patch_thr]]
                
            patchlist.extend(samplelist)
=======
            patchlist.extend(patient2patchlist[row.patientId])
>>>>>>> e14f4888d1e4228c257149865d6deb152971c162
        
        if tag == 'puretrain':
            patchlist.extend(jfsw_pos_patchdata)

        multilabel_jsondata = {
            "metainfo": {"classes": POSITIVE_CLASS},
            "data_list": []
        }
        binarylabel_txtdata = []

        print(f'Format {tag} patchlist to mmcls...')
        for patchinfo in tqdm(patchlist, ncols=80):
            imgname = f"{patchinfo['prefix']}/{patchinfo['filename']}"
            if patchinfo['prefix'] != 'partial_pos':
<<<<<<< HEAD
                clsids = []
                for i in patchinfo['clsnames']:
                    if i == 'SCC':
                        i = 'HSIL'
                    clsids.append(POSITIVE_CLASS.index(i))

=======
                clsids = [POSITIVE_CLASS.index(i) for i in patchinfo['clsnames']]
>>>>>>> e14f4888d1e4228c257149865d6deb152971c162
                multilabel_jsondata['data_list'].append({
                    "img_path": imgname,
                    "gt_label": list(set(clsids))
                })
<<<<<<< HEAD
                multilabel_pn_cnt[patchinfo['diagnose']] += 1

            binarylabel_txtdata.append(f'{imgname} {patchinfo["diagnose"]}\n')
            binary_pn_cnt[patchinfo['diagnose']] += 1
        
        print(f'{tag} multilabel_pn_cnt: {multilabel_pn_cnt}')
        print(f'{tag} binary_pn_cnt: {binary_pn_cnt}')

        if tag == 'puretrain' and neg_patch_thr > 0:
            tag += f'_npt{neg_patch_thr}'
=======
                

            binarylabel_txtdata.append(f'{imgname} {patchinfo["diagnose"]}\n')
        
>>>>>>> e14f4888d1e4228c257149865d6deb152971c162
        with open(f'{ann_dir}/multilabel_{tag}.json', 'w', encoding='utf-8') as f:
            json.dump(multilabel_jsondata, f, ensure_ascii=False)
        with open(f'{ann_dir}/binarylabel_{tag}.txt', 'w', encoding='utf-8') as f:
            f.writelines(binarylabel_txtdata)
<<<<<<< HEAD
        
=======

>>>>>>> e14f4888d1e4228c257149865d6deb152971c162

if __name__ == "__main__":
    ann_dir = f'{data_root}/annofiles'
    os.makedirs(ann_dir, exist_ok=True, mode=0o777)
<<<<<<< HEAD
    # partial_pos 样本只会用于阴阳二分类,不会用于多标签分类
    # neg_patch_thr 只会作用于训练集，验证集保持不变（真实情况就是阳性 patch 远少于阴性 patch）
    main()

'''
WS = 850
neg_patch_thr = -1
puretrain multilabel_pn_cnt: [51450, 12607]
val multilabel_pn_cnt: [26437, 6584]

puretrain binary_pn_cnt: [51450, 47883]
val binary_pn_cnt: [26437, 6616]

neg_patch_thr = 5
puretrain multilabel_pn_cnt: [5481, 12607]
puretrain binary_pn_cnt: [5481, 47883]


WS = 1600
neg_patch_thr = -1
puretrain multilabel_pn_cnt: [12814, 7801]
val multilabel_pn_cnt: [6214, 3916]

puretrain binary_pn_cnt: [12814, 21030]
val binary_pn_cnt: [6214, 3927]

neg_patch_thr = 3
puretrain multilabel_pn_cnt: [2455, 7801]
puretrain binary_pn_cnt: [2455, 21030]
'''
=======
    # partial_pos 样本只会用于阴阳二分类
    main()


>>>>>>> e14f4888d1e4228c257149865d6deb152971c162
    