import os
import json
import pandas as pd
from collections import defaultdict
import numpy as np
from pycocotools import mask as mask_utils
from tqdm import tqdm
import glob
from scipy import sparse
from prettytable import PrettyTable
from PIL import Image


POSITIVE_CLASS = ['abnormal']
CLASS_COLORS = [[139,0,139]]
data_root = 'data_resource/HMCHH'


def main(csvfiles_dir, imgname2ann):

    CV_nums = 5
    tag_list = []
    for i in range(CV_nums):
        folddir = f'{csvfiles_dir}/fold{i+1}'
        for tag in ['train', 'val']:
            tag_list.append({
                'name': f'fold{i+1}_{tag}',
                'csvpath': f'{folddir}/{tag}.csv'
            })
    tag_list.append({
        'name': 'test',
        'csvpath': f'{csvfiles_dir}/test.csv'
    })

    for tagitem in tag_list:
        df_data = pd.read_csv(tagitem['csvpath'])
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
        for row in tqdm(df_data.itertuples(index=True), total=len(df_data), ncols=80):
            imgname = os.path.basename(row.image_path)
            imgpath = f'{data_root}/JPEGImages/{imgname}'
            img = Image.open(imgpath)
            w,h = img.size
            rect_items = imgname2ann[imgname]
            # 按照bbox area 从大到小排序
            rect_items = sorted(
                rect_items,
                key=lambda annitem: (annitem['region'][2] - annitem['region'][0]) * (annitem['region'][3] - annitem['region'][1]),
                reverse=True  # 从大到小排序
            )
            pInfo = {
                'id': row.Index, 
                'width':w, 'height': h,
                'file_name': imgname,
                'diagnose': 1,
                'extra_info': {
                    'prefix': '',
                    'square_coords': [0,0,w,h]
                }
            }
            format_result['images'].append(pInfo)

            purename = pInfo['file_name'].split('.')[0]
            loader = np.load(f"{npz_mask_save_dir}/{purename}.npz")
            sparse_mask = sparse.coo_matrix((loader['data'], (loader['row'], loader['col'])), shape=loader['shape'])
            roi_mask = sparse_mask.toarray().astype(np.int16)
            annid_list = [child['annid'] for child in rect_items]

            for ann in rect_items:
                x1,y1,x2,y2 = ann['region']
                w,h = x2-x1, y2-y1
                clsname = ann['sub_class']
                inst_id = annid_list.index(ann['annid']) + 1
                annmask = roi_mask == inst_id
                rle = mask_utils.encode(np.asfortranarray(annmask))
                rle['counts'] = rle['counts'].decode('utf-8')
                format_result['annotations'].append({
                    "id": annid,
                    "image_id": row.Index,
                    "category_id": POSITIVE_CLASS.index(clsname) + 1,
                    "bbox": [x1,y1,w,h],
                    "segmentation": rle,
                    "area": w*h,
                    "iscrowd": 0,
                })
                annid += 1
            
        with open(f'{data_root}/annofiles_roi/{tagitem["name"]}.json', 'w', encoding='utf-8') as f:
            json.dump(format_result, f, ensure_ascii=False)


def statistic():
    CV_nums = 5
    foldtags = [f'fold{i+1}_{tag}' for tag in ['train', 'val'] for i in range(CV_nums)]
    foldtags.append('test')
    txt_lines = []
    for tag in foldtags:
        txt_lines.append(f'{"-"*20}{tag}{"-"*20}\n')
        with open(f'{ann_dir}/{tag}.json', 'r', encoding='utf-8') as f:
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

def format_imgname2ann(roi_data):
    result = {}
    for roiItem in roi_data:
        roi_annid = roiItem['annotations'][0]['annid']
        imagname = f"{roiItem['patientId']}_{roi_annid}.png"
        result[imagname] = roiItem['annotations'][0]['children']
    return result

if __name__ == "__main__":
    ann_dir = f'{data_root}/annofiles_roi'
    os.makedirs(ann_dir, exist_ok=True, mode=0o777)
    npz_mask_save_dir = 'data_resource/HMCHH/roi_inst_mask'

    csvfiles_dir = f'{data_root}/csvfiles'
    with open('data_resource/HMCHH/annofiles_roi/unify_ann.json', 'r', encoding='utf-8') as f:
        roi_data = json.load(f)
    imgname2ann = format_imgname2ann(roi_data)
    main(csvfiles_dir, imgname2ann)
    statistic()
    # clear_imgs()

