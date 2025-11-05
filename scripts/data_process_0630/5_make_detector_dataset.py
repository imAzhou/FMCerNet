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
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pycocotools.coco import COCO

WINDOW_SIZE = 800
POSITIVE_CLASS = ['AGC', 'ASC-US','LSIL', 'ASC-H', 'HSIL']
CLASS_COLORS = [[31,119,180], [255,153,153], [255,105,180], [255,20,147], [139,0,139]]
data_root = f'data_resource/0630/WINDOW_SIZE_{WINDOW_SIZE}'

def coco_format(patchlist):
    format_result = {
        'categories': [{
            'id': idx+1,
            'name': clsname,
            'color': clscolor,
        } for idx, clsname,clscolor in zip(range(len(POSITIVE_CLASS)), POSITIVE_CLASS, CLASS_COLORS)],
        'images': [],
        'annotations': [],
        'info': {}
    }

    annid = 0
    for idx,pInfo in enumerate(tqdm(patchlist, ncols=80)):
        format_result['images'].append(
            {'id': idx, 'width': WINDOW_SIZE, 'height': WINDOW_SIZE,
             'file_name': f"{pInfo['prefix']}/{pInfo['filename']}", 
             'extra_info': {
                    'prefix': pInfo['prefix'],
                    'square_coords': pInfo['square_coords']
                },
             'diagnose': pInfo['diagnose']})
        
        inst_mask = np.load(f"{data_root}/patch_inst_mask/{pInfo['maskfile']}")['patch_mask']
        for bbox,clsname,inst_id in zip(pInfo['bboxes'],pInfo['clsnames'],range(len(pInfo['bboxes']))):
            x1,y1,x2,y2 = bbox
            w,h = x2-x1, y2-y1
            annmask = inst_mask == inst_id+1
            rle = mask_utils.encode(np.asfortranarray(annmask))
            rle['counts'] = rle['counts'].decode('utf-8')
            if clsname == 'SCC':
                clsname = 'HSIL'
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

def main():
    with open(f'{data_root}/ann_jsons/patches_in_RoI_pure_valid.json', 'r', encoding='utf-8') as f:
        RoI_patchlist = json.load(f)

    patient2patchlist = defaultdict(list)
    for patchInfo in RoI_patchlist:
        if patchInfo['prefix'] == 'total_pos':
            patient2patchlist[patchInfo['patientId']].append(patchInfo)
    
    data_group = {
        'puretrain': 'data_resource/0630/4_pure_train.csv',
        'val': 'data_resource/0630/6_val.csv'
    }
    for tag,csvpath in data_group.items():
        df_data = pd.read_csv(csvpath)
        patchlist = []
        for row in tqdm(df_data.itertuples(index=False), total=len(df_data), ncols=80):
            patchlist.extend(patient2patchlist[row.patientId])
            
        patchInCOCO = coco_format(patchlist)
        savename = f'{tag}_cocoformat'
        with open(f'{data_root}/annofiles/{savename}.json', 'w', encoding='utf-8') as f:
            json.dump(patchInCOCO, f, ensure_ascii=False)
    
    tags = data_group.keys()
    return tags
    
   
def statistic(tags):
    txt_lines = []
    for tag in tags:
    # for tag in ['puretrain']:
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
                if len(annoInimg[imginfo['id']]) == 81:
                    print()
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

def draw_dataset_gt():
    jsonfile = f'{data_root}/annofiles/puretrain_cocoformat.json'
    with open(jsonfile, 'r', encoding='utf-8') as f:
        patch_COCOinfo = json.load(f)
    coco = COCO(jsonfile)

    visual_cnt, max_cnt = 0, 50
    random.shuffle(patch_COCOinfo['images'])
    for imgitem in tqdm(patch_COCOinfo['images'], ncols=80):
        if visual_cnt > max_cnt:
            break
        annids = coco.getAnnIds([imgitem['id']])
        annos = coco.loadAnns(annids)
        filename = imgitem['file_name'].split('/')[-1]
        # if filename != 'WXL_1_25_1717869845996_0.png':
        #     continue
        
        img = Image.open(f'{data_root}/images/{imgitem["file_name"]}')
        fig = plt.figure(figsize=(13,13))
        ax = fig.add_subplot(111)
        ax.imshow(img)

        for anninfo in annos:
            cls_color = CLASS_COLORS[anninfo['category_id']-1]
            edgecolor = np.array([cls_color[0]/255, cls_color[1]/255, cls_color[2]/255, 1])
            cls_name = POSITIVE_CLASS[anninfo['category_id']-1]
            x, y, w, h = anninfo['bbox']
            ax.add_patch(plt.Rectangle((x, y), w, h, edgecolor=edgecolor, facecolor=(0,0,0,0), lw=2))
            ax.text(x, y, cls_name, fontsize=10, color='white',
                bbox=dict(facecolor=np.array(cls_color)/255., alpha=0.5, edgecolor='none'))

        ax.set_title('GT info')
        ax.set_axis_off()
        patches = [mpatches.Patch(facecolor=np.array(CLASS_COLORS[i])/255., label=POSITIVE_CLASS[i], edgecolor='black') for i in range(len(POSITIVE_CLASS))]
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='large')

        plt.tight_layout()
        savedir = f'statistic_results/visual_results/gt_{WINDOW_SIZE}'
        os.makedirs(savedir, exist_ok=True, mode=0o777)
        plt.savefig(f'{savedir}/{filename}')
        plt.close()
        visual_cnt += 1



if __name__ == "__main__":
    ann_dir = f'{data_root}/annofiles'
    os.makedirs(ann_dir, exist_ok=True, mode=0o777)
    
    tags = main()
    statistic(tags)

    # draw_dataset_gt()

    