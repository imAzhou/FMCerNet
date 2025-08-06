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

WINDOW_SIZE = 1600
POSITIVE_CLASS = ['AGC', 'ASC-US','LSIL', 'ASC-H', 'HSIL']
CLASS_COLORS = [[31,119,180], [255,153,153], [255,105,180], [255,20,147], [139,0,139]]
<<<<<<< HEAD
data_root = f'data_resource/0630/WINDOW_SIZE_{WINDOW_SIZE}'
=======
data_root = f'data_resource/WINDOW_SIZE_{WINDOW_SIZE}'
>>>>>>> e14f4888d1e4228c257149865d6deb152971c162
neg_patch_thr = 0

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
             'extra_info': {
                    'prefix': pInfo['prefix'],
                    'square_coords': pInfo['square_coords']
                },
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
    # with open(f'data_resource/0630/WINDOW_SIZE_{WINDOW_SIZE}/ann_jsons/ppatches_in_NegSlide.json', 'r', encoding='utf-8') as f:
    #     negslide_patchlist = json.load(f)
    with open(f'{data_root}/ann_jsons/patches_in_RoI_pure_valid.json', 'r', encoding='utf-8') as f:
        RoI_patchlist = json.load(f)
    # RoI_patchlist = ensure_exist(RoI_patchlist)
    RoI_patchlist = filter_slide_neg(RoI_patchlist, neg_patch_thr=neg_patch_thr) # 控制每张病人切片的阴性 patch 数量

    patient2patchlist = defaultdict(list)
    # for patchInfo in [*negslide_patchlist, *RoI_patchlist]:
    for patchInfo in RoI_patchlist:
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
        savename = f'{tag}_noNeg_cocoformat' if neg_patch_thr==0 else f'{tag}_cocoformat'
        with open(f'{data_root}/annofiles/{savename}.json', 'w', encoding='utf-8') as f:
            json.dump(patchInCOCO, f, ensure_ascii=False)
    
        if tag == 'puretrain' and use_jfsw:
            # with open(f'{data_root}/ann_jsons/jfswtrain_patches_in_NegSlide.json', 'r', encoding='utf-8') as f:
            #     jfswtrain_negslide_patchlist = json.load(f)
            with open(f'{data_root}/ann_jsons/patches_in_RoI_jfsw_valid.json', 'r', encoding='utf-8') as f:
                jfsw_pos_patchdata = json.load(f)
            df_jfswtrain = pd.read_csv('data_resource/0630/5_jfsw_train.csv')
            # jfsw_patchdata = [*jfswtrain_negslide_patchlist, *jfsw_pos_patchdata]
            jfsw_patchdata = jfsw_pos_patchdata
            jfsw_patchdata = [i for i in jfsw_patchdata if i['patientId'] in list(df_jfswtrain['patientId'])]

            fusionPatchInCOCO = coco_format([*patchlist, *jfsw_patchdata])
            savename = f'fusiontrain_noNeg_cocoformat' if neg_patch_thr==0 else f'fusiontrain_cocoformat'
            with open(f'{data_root}/annofiles/{savename}.json', 'w', encoding='utf-8') as f:
                json.dump(fusionPatchInCOCO, f, ensure_ascii=False)
        
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

<<<<<<< HEAD
def draw_dataset_gt():
    jsonfile = f'{data_root}/annofiles/fusiontrain_noNeg_cocoformat.json'
    with open(jsonfile, 'r', encoding='utf-8') as f:
        patch_COCOinfo = json.load(f)
    coco = COCO(jsonfile)

    for imgitem in tqdm(patch_COCOinfo['images'], ncols=80):
        annids = coco.getAnnIds([imgitem['id']])
        annos = coco.loadAnns(annids)
        filename = imgitem['file_name'].split('/')[-1]
        if filename != 'WXL_1_25_1717869845996_0.png':
            continue
        
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

=======
>>>>>>> e14f4888d1e4228c257149865d6deb152971c162
def reset_scc2hsil(tags):
    for tag in tags:
        jsonfile = f'{data_root}/annofiles/{tag}_cocoformat.json'
        with open(jsonfile, 'r', encoding='utf-8') as f:
            patch_COCOinfo = json.load(f)
        
        annotations = []
        for annoinfo in patch_COCOinfo['annotations']:
            if annoinfo['category_id'] == 6:
                annoinfo['category_id'] = 5
            annotations.append(annoinfo)
        
        patch_COCOinfo['annotations'] = annotations
        patch_COCOinfo['categories'] = [i for i in patch_COCOinfo['categories'] if i['id']!=6]
        with open(jsonfile, 'w', encoding='utf-8') as f:
            json.dump(patch_COCOinfo, f, ensure_ascii=False)

if __name__ == "__main__":
    ann_dir = f'{data_root}/annofiles'
    os.makedirs(ann_dir, exist_ok=True, mode=0o777)
    
    # main(use_jfsw=True)
    tags = [
        # 'fusiontrain', 'puretrain', 'val',
        'fusiontrain_noNeg', 'puretrain_noNeg', 'val_noNeg',
        # 'puretrain_aug', 'puretrain_withneg', 'puretrain_aug_withneg'
        ]
    # reset_scc2hsil(tags)
<<<<<<< HEAD
    # statistic(tags)
=======
    statistic(tags)
>>>>>>> e14f4888d1e4228c257149865d6deb152971c162
    # clear_imgs()
    draw_dataset_gt()

    