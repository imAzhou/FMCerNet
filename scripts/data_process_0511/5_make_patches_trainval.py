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

WINDOW_SIZE = 750
POSITIVE_CLASS = ['AGC', 'ASC-US','LSIL', 'ASC-H', 'HSIL']
CLASS_COLORS = [[31,119,180], [255,153,153], [255,105,180], [255,20,147], [139,0,139]]
# data_root = '/c22073/zly/datasets/CervicalDatasets/LCerScanv1_750'
data_root = 'data_resource/0511/WINDOW_SIZE_750'


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

    annid = 0
    for idx,pInfo in enumerate(tqdm(patchlist, ncols=80)):
        format_result['images'].append(
            {'id': idx, 'width': WINDOW_SIZE, 'height': WINDOW_SIZE,
             'file_name': f"{pInfo['prefix']}/{pInfo['filename']}", 
             'prefix': pInfo['prefix'], 
             'diagnose': pInfo['diagnose']})
        
        if pInfo['diagnose'] == 1:
            inst_mask = np.load(f"{data_root}/patch_inst_mask/{pInfo['maskfile']}")['patch_mask']
            for bbox,clsname,inst_id in zip(pInfo['bboxes'],pInfo['clsnames'],range(len(pInfo['bboxes']))):
                x1,y1,x2,y2 = bbox
                w,h = x2-x1, y2-y1
                annmask = inst_mask == inst_id+1
                rle = mask_utils.encode(np.asfortranarray(annmask))
                rle['counts'] = rle['counts'].decode('utf-8')
                format_result['annotations'].append({
                    "id": annid,
                    "image_id": idx,
                    "category_id": POSITIVE_CLASS.index(clsname) + 1,
                    "segmentation": rle,
                    "bbox": [x1,y1,w,h],
                    "area": w*h,
                    "iscrowd": 0,
                })
                annid += 1

    return format_result

def ensure_exist(RoI_patchlist):
    new_RoI_patchlist = []
    for pInfo in tqdm(RoI_patchlist, ncols=90, desc="Ensuring patch file in RoI exist"):
        if os.path.exists(f'{data_root}/images/{pInfo["prefix"]}/{pInfo["filename"]}'):
            new_RoI_patchlist.append(pInfo)
    return new_RoI_patchlist

def main(use_jfsw):
    with open('data_resource/0511/WINDOW_SIZE_750/ann_jsons/ppatches_in_NegSlide.json', 'r', encoding='utf-8') as f:
        negslide_patchlist = json.load(f)
    with open(f'{data_root}/ann_jsons/patches_in_RoI_pure_valid.json', 'r', encoding='utf-8') as f:
        RoI_patchlist = json.load(f)
    RoI_patchlist = ensure_exist(RoI_patchlist)
    RoI_patchlist = filter_slide_neg(RoI_patchlist, neg_patch_thr=300) # 控制每张病人切片的阴性 patch 数量

    patient2patchlist = defaultdict(list)
    for patchInfo in [*negslide_patchlist, *RoI_patchlist]:
    # for patchInfo in RoI_patchlist:
        patient2patchlist[patchInfo['patientId']].append(patchInfo)
    
    data_group = {
        'puretrain': 'data_resource/0511/4_pure_train.csv',
        # 'val': 'data_resource/0511/6_val.csv'
    }
    for tag,csvpath in data_group.items():
        df_data = pd.read_csv(csvpath)
        patchlist = []
        for row in tqdm(df_data.itertuples(index=False), total=len(df_data), ncols=80):
            patchlist.extend(patient2patchlist[row.patientId])
        patchInCOCO = coco_format(patchlist)
        
        with open(f'{data_root}/annofiles/{tag}_cocoformat_new.json', 'w', encoding='utf-8') as f:
            json.dump(patchInCOCO, f, ensure_ascii=False)
    
        if tag == 'puretrain' and use_jfsw:
            with open('data_resource/0511/WINDOW_SIZE_750/ann_jsons/jfswtrain_patches_in_NegSlide.json', 'r', encoding='utf-8') as f:
                jfswtrain_negslide_patchlist = json.load(f)
            with open(f'{data_root}/ann_jsons/patches_in_RoI_jfsw_valid.json', 'r', encoding='utf-8') as f:
                jfsw_pos_patchdata = json.load(f)
            df_jfswtrain = pd.read_csv('data_resource/0511/5_jfsw_train.csv')
            jfsw_patchdata = [*jfswtrain_negslide_patchlist, *jfsw_pos_patchdata]
            jfsw_patchdata = [i for i in jfsw_patchdata if i['patientId'] in list(df_jfswtrain['patientId'])]

            fusionPatchInCOCO = coco_format([*patchlist, *jfsw_patchdata])
            with open(f'{data_root}/annofiles/fusiontrain_cocoformat_new.json', 'w', encoding='utf-8') as f:
                json.dump(fusionPatchInCOCO, f, ensure_ascii=False)
        

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
    txt_lines = []
    for tag in ['fusiontrain', 'puretrain', 'val']:
        txt_lines.append(f'{"-"*30}{tag}{"-"*30}\n')
        with open(f'{data_root}/annofiles/{tag}_cocoformat.json', 'r', encoding='utf-8') as f:
            patch_COCOinfo = json.load(f)
        annoInimg = defaultdict(list)
        bbox_cls_cnt = [0]*len(POSITIVE_CLASS)
        for annoinfo in patch_COCOinfo['annotations']:
            annoInimg[annoinfo['image_id']].append(annoinfo)
            bbox_cls_cnt[annoinfo['category_id']-1] += 1
        result_table = PrettyTable(title=f'{tag} BBox Info')
        result_table.field_names = POSITIVE_CLASS + ['Sum']
        result_table.add_row(bbox_cls_cnt + [sum(bbox_cls_cnt)])
        print(result_table)
        txt_lines.append(str(result_table))
        txt_lines.append('\n')

        pn_cnt = [0,0]
        consist_error = [0,0]
        pos_bbox_cnt = defaultdict(int)
        for imginfo in tqdm(patch_COCOinfo['images'], ncols=80):
            pn_cnt[imginfo['diagnose']] += 1
            if imginfo['diagnose'] == 1 and len(annoInimg[imginfo['id']]) == 0:
                consist_error[1] += 1
            elif imginfo['diagnose'] == 0 and len(annoInimg[imginfo['id']]) != 0:
                consist_error[0] += 1
            if imginfo['diagnose'] == 1:
                pos_bbox_cnt[imginfo['id']] = len(annoInimg[imginfo['id']])
        if sum(consist_error) != 0:
            print(f'ERROR: consist_error {consist_error}')
        
        mean = sum(pos_bbox_cnt.values()) / len(pos_bbox_cnt.values())
        minv,maxv = min(pos_bbox_cnt.values()),max(pos_bbox_cnt.values())

        result_table = PrettyTable(title=f'{tag} Image Info')
        result_table.field_names = ['Neg', 'Pos', 'Sum', 'Pos bbox avg/min/max']
        result_table.add_row(pn_cnt + [sum(pn_cnt), f'{mean:.2}/{minv}/{maxv}'])
        print(result_table)
        txt_lines.append(str(result_table))
        txt_lines.append('\n\n')
    with open(f'{ann_dir}/statistic_result.txt', 'w') as f:
        f.writelines(txt_lines)

def clear_imgs():
    keep_filename = []
    for tag in ['fusiontrain','puretrain','val']:
        with open(f'{ann_dir}/{tag}_cocoformat_new.json', 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        for patchinfo in tqdm(json_data['images'], ncols=80):
            keep_filename.append(patchinfo["file_name"].split('/')[1])
            if not os.path.exists(f'{data_root}/images/{patchinfo["file_name"]}'):
                print('ERROR: path not exist!')
    #         pn_cnt[patchinfo['diagnose']] += 1
    #     print(f'{tag}: {len(json_data["images"])}, [neg, pos]: [{pn_cnt[0]}, {pn_cnt[1]}]')
    
    exists_imgpath = glob.glob(f'{data_root}/images/**/*.png')
    print(f'keep_filename nums: {len(set(keep_filename))}')
    print(f'exists_imgpath nums: {len(exists_imgpath)}')

    # for imgpath in tqdm(exists_imgpath, ncols=80):
    #     filename = os.path.basename(imgpath)
    #     if filename not in keep_filename:
    #         os.remove(imgpath)



if __name__ == "__main__":
    ann_dir = f'{data_root}/annofiles'
    os.makedirs(ann_dir, exist_ok=True, mode=0o777)
    
    # main(use_jfsw=True)
    statistic()
    # clear_imgs()

    