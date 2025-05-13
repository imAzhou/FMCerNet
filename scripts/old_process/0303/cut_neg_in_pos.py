import pandas as pd
from tqdm import tqdm
from PIL import Image
import os
import json
from prettytable import PrettyTable
from cerwsi.utils import (KFBSlide,remap_points,read_json_anno,is_bbox_inside,random_cut_square)

POSITIVE_CLASS = ['ASC-US', 'LSIL', 'ASC-H', 'HSIL', 'SCC', 'AGC-NOS', 'AGC', 'AGC-N', 'AGC-FN']

def overlap_enough(bbox1, bbox2, min_overlap):
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    return inter_width > min_overlap and inter_height > min_overlap

def filter_pos_bbox(annos):
    all_pos_bbox = []
    for ann_ in annos:
        ann = remap_points(ann_)
        if ann is None:
            continue
        sub_class = ann.get('sub_class')
        region = ann.get('region')
        x,y = region['x'],region['y']
        w,h = region['width'],region['height']
        if w >50 and h>50 and sub_class in POSITIVE_CLASS:
            all_pos_bbox.append([x, y, x+w, y+h])
    return all_pos_bbox

def ensure_neg(square, pos_bbox):
    for pbox in pos_bbox:
        if is_bbox_inside(pbox, square, tolerance=20):
            return False
        if overlap_enough(pbox, square, min_overlap=50):
            return False
    return True

def cut_square(mode, df_data: pd.DataFrame):
    '''
    从 ROI 中随机裁剪 10 张正方形 patch，
    过滤出阴性 patch (内不含阳性标注，与所有阳性标注交集不超过 50 px)
    '''
    df_JF1 = pd.read_csv('data_resource/cls_pn/group_csv/JFSW_1.csv')
    df_JF2 = pd.read_csv('data_resource/cls_pn/group_csv/JFSW_2.csv')
    df_JF = pd.concat([df_JF1, df_JF2], ignore_index=True)

    save_dir = f'data_resource/0103/images/Neg_in_PosSlide'
    os.makedirs(save_dir, exist_ok=True)
    slide_cnt,totoal_img_cnt = 0,0
    total_items = []
    for row in tqdm(df_data.itertuples(index=True), total=len(df_data), ncols=80):
        if row.kfb_clsid == 0 or 'JFSW' not in row.kfb_source:
            continue
        patient_row = df_JF.loc[df_JF['patientId'] == row.patientId].iloc[0]
        json_path = f'{data_root_dir}/{patient_row.json_path}'
        annos = read_json_anno(json_path)
        pos_bbox = filter_pos_bbox(annos)

        slide = KFBSlide(f'{data_root_dir}/{row.kfb_path}')
        slide_item = {
            'patientId': row.patientId,
            'kfb_path': row.kfb_path,
            'patch_list': []
        }
        for ann_ in annos:
            ann = remap_points(ann_)
            if ann is None:
                continue
            sub_class = ann.get('sub_class')
            region = ann.get('region')
            x,y = region['x'],region['y']
            w,h = region['width'],region['height']
            if w >700 and h>700 and sub_class == 'ROI':
                x1,y1 = random_cut_square((x,y,w,h), 700)
                square = [x1,y1,x1+700,y1+700]
                if ensure_neg(square, pos_bbox):
                    filename = f"{row.patientId}_{len(slide_item['patch_list'])}.png"
                    location, level, size = (x1,y1), 0, (700, 700)
                    read_result = Image.fromarray(slide.read_region(location, level, size))
                    read_result.save(f'{save_dir}/{filename}')
                    slide_item['patch_list'].append({
                        'filename': filename,
                        'square_x1y1': [x1,y1],
                        'bboxes': [],
                        'clsnames': [],
                        'diagnose': 0,
                        'gtmap_14': []
                    })

        if len(slide_item['patch_list']) > 0:
            slide_cnt += 1
            totoal_img_cnt += len(slide_item['patch_list'])
            total_items.append(slide_item)

    print(f'mode {mode} slide: {slide_cnt}/{len(df_data)}')
    print(f'mode {mode} patch: {totoal_img_cnt}')
    return total_items

if __name__ == '__main__':
    data_root_dir = '/medical-data/data'

    for mode in ['train', 'val']:
        df_data = pd.read_csv(f'data_resource/0103/annofiles/1223_{mode}.csv')
        anno_jsonpath = f'data_resource/0103/annofiles/{mode}_neg_in_posslide_patches.json'
        total_items = cut_square(mode, df_data)
        with open(anno_jsonpath, 'w') as f:
            json.dump(total_items, f)

'''
mode train: 9596
mode val: 2320
'''