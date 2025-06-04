import os
import json
import pandas as pd
from collections import defaultdict
import numpy as np
from pycocotools import mask as mask_utils
from tqdm import tqdm
import glob
from prettytable import PrettyTable
from PIL import Image


POSITIVE_CLASS = ['abnormal']
CLASS_COLORS = [[139,0,139]]
# data_root = '/c22073/zly/datasets/CervicalDatasets/HMCHH'
data_root = 'data_resource/HMCHH'


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
        pInfo['id'] = idx
        annos = pInfo['annotations']
        del pInfo['annotations']
        format_result['images'].append(pInfo)
        
        for ann in annos:
            x1,y1,x2,y2 = ann['region']
            w,h = x2-x1, y2-y1
            clsname = ann['sub_class']
            format_result['annotations'].append({
                "id": annid,
                "image_id": idx,
                "category_id": POSITIVE_CLASS.index(clsname) + 1,
                "bbox": [x1,y1,w,h],
                "area": w*h,
                "iscrowd": 0,
            })
            annid += 1

    return format_result

def parse_annos(annotations):
    rect_items = []
    for anno in annotations.split(';'):
        x,y = [],[]
        anno = anno[2:]  # one box coord str
        anno = anno.split(" ")
        for i in range(len(anno)):
            if i % 2 == 0:
                x.append(float(anno[i]))
            else:
                y.append(float(anno[i]))

        xmin,xmax = min(x),max(x)
        ymin,ymax = min(y),max(y)
        rect_items.append(dict(
            sub_class='abnormal', region=[xmin, ymin, xmax, ymax],
        ))
    
    return rect_items

def main(csvfiles_dir):

    CV_nums = 5
    for i in range(CV_nums):
        folddir = f'{csvfiles_dir}/fold{i+1}'
        for tag in ['train', 'val']:
            df_data = pd.read_csv(f'{folddir}/{tag}.csv')
            
            patchlist = []
            for row in tqdm(df_data.itertuples(index=False), total=len(df_data), ncols=80):
                imgname = os.path.basename(row.image_path)
                imgpath = f'{data_root}/JPEGImages/{imgname}'
                img = Image.open(imgpath)
                w,h = img.size
                rect_items = parse_annos(row.annotation)
                patchlist.append({
                    'width':w, 'height': h,
                    'file_name': imgname,
                    'diagnose': 1,
                    'annotations': rect_items
                })
                
            patchInCOCO = coco_format(patchlist)
            
            with open(f'{data_root}/annofiles_roi/fold{i+1}_{tag}_cocoformat.json', 'w', encoding='utf-8') as f:
                json.dump(patchInCOCO, f, ensure_ascii=False)
    
    df_test = pd.read_csv(f'{csvfiles_dir}/test.csv')
    patchlist = []
    for row in tqdm(df_test.itertuples(index=False), total=len(df_test), ncols=80):
        imgname = os.path.basename(row.image_path)
        imgpath = f'{data_root}/JPEGImages/{imgname}'
        img = Image.open(imgpath)
        w,h = img.size
        rect_items = parse_annos(row.annotation)
        patchlist.append({
            'width':w, 'height': h,
            'diagnose': 1,
            'file_name': imgname,
            'annotations': rect_items
        })
    patchInCOCO = coco_format(patchlist)
    
    with open(f'{ann_dir}/test_cocoformat.json', 'w', encoding='utf-8') as f:
        json.dump(patchInCOCO, f, ensure_ascii=False)
        


def statistic():
    CV_nums = 5
    foldtags = [f'fold{i+1}_{tag}' for tag in ['train', 'val'] for i in range(CV_nums)]
    foldtags.append('test')
    txt_lines = []
    for tag in foldtags:
        txt_lines.append(f'{"-"*20}{tag}{"-"*20}\n')
        with open(f'{ann_dir}/{tag}_cocoformat.json', 'r', encoding='utf-8') as f:
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
        with open(f'{ann_dir}/{tag}_cocoformat.json', 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        for patchinfo in tqdm(json_data['images'], ncols=80):
            keep_filename.append(patchinfo["file_name"].split('/')[1])
            # if not os.path.exists(f'{data_root}/images/{patchinfo["file_name"]}'):
            #     print('ERROR: path not exist!')
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
    ann_dir = f'{data_root}/annofiles_roi'
    os.makedirs(ann_dir, exist_ok=True, mode=0o777)
    
    csvfiles_dir = f'{data_root}/csvfiles'
    # main(csvfiles_dir)
    statistic()
    # clear_imgs()

