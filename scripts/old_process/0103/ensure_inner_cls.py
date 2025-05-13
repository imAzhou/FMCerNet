import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import json
import glob
import random
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from cerwsi.utils import KFBSlide,read_json_anno,is_bbox_inside,random_cut_square,remap_points

def get_RoI_info():
    df_jf1 = pd.read_csv('data_resource/cls_pn/group_csv/JFSW_1.csv')
    df_jf2 = pd.read_csv('data_resource/cls_pn/group_csv/JFSW_2.csv')
    df_jf = pd.concat([df_jf1, df_jf2], ignore_index=True)
    

    record_kfb = [0,0]
    for row in tqdm(df_jf.itertuples(index=True), total=len(df_jf), ncols=80):

        if not isinstance(row.json_path, str):
            continue

        json_path = f'{data_root_dir}/{row.json_path}'
        annos = read_json_anno(json_path)
        filter_roi = dict(roi_list = [], cut_window_size = WINDOW_SIZE)
        ROI_items = [ann for ann in annos if ann['sub_class'] == 'ROI']
        for item_ in ROI_items:
            roi_item = remap_points(item_)
            if roi_item is None:
                continue
            region = roi_item.get('region')
            x,y = region['x'],region['y']
            w,h = region['width'],region['height']
            if w > WINDOW_SIZE and h > WINDOW_SIZE:
                ROI_inside = get_ROI_inside([x,y,w,h], annos)
                if len(ROI_inside) > 0:
                    roi_item_dict = square_cut_roi([x,y,w,h], ROI_inside)
                    filter_roi['roi_list'].append(roi_item_dict)
        
        if len(filter_roi['roi_list']) == 0:
            no_roi_list.append(f'{row.kfb_path}\n')
        else:
            idx = 0 if row.kfb_clsname == 'NILM' else 1
            record_kfb[idx] += 1
            with open(f'{json_savedir}/{row.patientId}.json', 'w') as f:
                json.dump(filter_roi, f)
    
    with open('statistic_results/jfsw_no_roi_v2.txt', 'w') as f:
        f.writelines(no_roi_list)
    
    print(f'total kfb nums: {sum(record_kfb)}, Neg: {record_kfb[0]}, Pos:{record_kfb[1]}')


if __name__ == '__main__':
    pass