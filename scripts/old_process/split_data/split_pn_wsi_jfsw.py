import json
from prettytable import PrettyTable
import os
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import chardet
from tqdm import tqdm
from PIL import Image
from cerwsi.utils import KFBSlide, is_bbox_inside, random_cut_fn

NEGATIVE_CLASS = ['NILM', 'GEC']
POSITIVE_CLASS = ['ASC-US', 'LSIL', 'ASC-H', 'HSIL', 'SCC', 'AGC-NOS', 'AGC', 'AGC-N', 'AGC-FN']
RANDOM_CUT_POSITIVE = True
save_dir = 'data_resource/cls_pn/jfsw_valid_ann'
os.makedirs(save_dir, exist_ok=True)
save_img_flag = False

def save_img_fn(bbox_info, slide:KFBSlide, filename, random_cut = False):
    sub_class = bbox_info[2]
    img_save_dir = f'{save_dir}/{sub_class}'
    save_img_path = f'{img_save_dir}/{sub_class}_{filename}'

    if save_img_flag:
        os.makedirs(img_save_dir, exist_ok=True)
        x1,y1,x2,y2 = bbox_info[1]
        w,h = x2-x1, y2-y1
        location, level, size = (x1,y1), 0, (w,h)
        read_result = Image.fromarray(slide.read_region(location, level, size))
        read_result.save(save_img_path)
        
    if random_cut and save_img_flag:
        cut_results = random_cut_fn(x1,y1,w,h)
        for rc_idx,new_rect in enumerate(cut_results):
            new_x1,new_y1,new_w,new_h = new_rect
            location, level, size = (new_x1,new_y1), 0, (new_w,new_h)
            read_result = Image.fromarray(slide.read_region(location, level, size))
            img_save_dir = f'{save_dir}/random_cut/{sub_class}'
            os.makedirs(img_save_dir, exist_ok=True)
            read_result.save(f'{img_save_dir}/{sub_class}_rc{rc_idx}_{filename}')

    return save_img_path

def read_json(json_path, slide:KFBSlide, patientId):
    '''
    return valid positive and negative bbox
    '''
    with open(json_path, 'rb') as f:
        result = chardet.detect(f.read())
        encoding = result['encoding']
    with open(json_path,'r',encoding = encoding) as f:
        data = json.load(f)
    annotations = data['annotation']
    max_x, max_y = slide.level_dimensions[0]

    pos_bbox_record, neg_bbox_record = [], []
    for idx,i in enumerate(annotations):
        region = i.get('region')
        sub_class = i.get('sub_class')
        w,h = region['width'],region['height']
        x1,y1 = region['x'],region['y']
        x2,y2 = x1+w, y1+h
        if x2 > max_x or y2 > max_y:
            continue
        if sub_class in NEGATIVE_CLASS and (w>224 and h>224):
            neg_bbox_record.append((idx, [x1,y1,x2,y2], sub_class))
        if sub_class in POSITIVE_CLASS and (w>100 and h>100):
            pos_bbox_record.append((idx, [x1,y1,x2,y2], sub_class))

    saved_img_info = []
    valid_idx = 0
    for bbox_info in neg_bbox_record:
        valid_flag = True
        for idx,i in enumerate(annotations):
            region = i.get('region')
            sub_class = i.get('sub_class')
            w,h = region['width'],region['height']
            x1,y1 = region['x'],region['y']
            x2,y2 = x1+w, y1+h
            if idx != bbox_info[0] and is_bbox_inside([x1,y1,x2,y2], bbox_info[1]) and sub_class not in NEGATIVE_CLASS:
                valid_flag = False
                break
        if valid_flag:
            filename = f'{patientId}_{valid_idx}.png'
            save_img_path = save_img_fn(bbox_info, slide, filename)
            valid_idx += 1
            saved_img_info.append([patientId, bbox_info[2], save_img_path])
    
    for bbox_info in pos_bbox_record:
        valid_flag = True
        for idx,i in enumerate(annotations):
            region = i.get('region')
            sub_class = i.get('sub_class')
            w,h = region['width'],region['height']
            x1,y1 = region['x'],region['y']
            x2,y2 = x1+w, y1+h
            if idx != bbox_info[0] and is_bbox_inside([x1,y1,x2,y2], bbox_info[1]) and sub_class != bbox_info[2]:
                valid_flag = False
                break
        if valid_flag:
            filename = f'{patientId}_{valid_idx}.png'
            save_img_path = save_img_fn(bbox_info, slide, filename, RANDOM_CUT_POSITIVE)
            valid_idx += 1
            saved_img_info.append([patientId, bbox_info[2], save_img_path])
    
    return saved_img_info

def cut_valid_JFSW_1():
    JFSW_1_root_dir = '/disk/medical_datasets/cervix/JFSW'
    json_dir_1 = f'{JFSW_1_root_dir}/阳性json'
    positive_diag = '/disk/medical_datasets/cervix/JFSW/JFSW病人整片分类.xlsx'
    df_positive_diag = pd.read_excel(positive_diag, sheet_name='阳性')

    kfb2patientId = []
    idx = 0
    total_saved_img_info = []
    for json_name in tqdm(os.listdir(json_dir_1), ncols=80):
        json_path = f'{json_dir_1}/{json_name}'
        kfb_path = f'{JFSW_1_root_dir}/阳性/{json_name}'.replace('.json','.kfb')
        filename = os.path.basename(kfb_path)
        filtered_rows = df_positive_diag[df_positive_diag['new_name'] == filename]
        assert len(filtered_rows) == 1
        kfb_clsname = filtered_rows.iloc[0]['diagnostic_type']
        slide = KFBSlide(kfb_path)
        patientId = f'JFSW_1_{idx}'
        saved_img_info = read_json(json_path, slide, patientId)
        kfb2patientId.append([patientId, kfb_clsname, len(saved_img_info), kfb_path])
        total_saved_img_info.extend(saved_img_info)
        idx += 1
    
    df = pd.DataFrame(kfb2patientId, columns = ['patientId', 'kfb_clsname', 'valid_img_cnt', 'kfb_path'])
    df.to_csv(f'{save_dir}/JFSW_1_kfb_info.csv', index=False)
    
    df = pd.DataFrame(total_saved_img_info, columns = ['patientId', 'img_clsname', 'save_img_path'])
    df.to_csv(f'{save_dir}/JFSW_1_img_info.csv', index=False)

def cut_valid_JFSW_2():
    root_dir = '/disk/medical_datasets/cervix/JFSW_1109'
    all_kfb_path = glob.glob(f'{root_dir}/**/*.kfb')
    kfb2patientId = []
    idx = 0
    total_saved_img_info = []
    for kfb_path in tqdm(all_kfb_path, ncols=80):
        path = Path(kfb_path)
        directories = path.parents
        filename = os.path.basename(kfb_path).replace('.kfb','.json')
        json_path = f'{directories[0]}/json/{filename}'
        if not os.path.exists(json_path):
            continue
        kfb_clsname = path.parts[-2]
        slide = KFBSlide(kfb_path)
        patientId = f'JFSW_2_{idx}'
        saved_img_info = read_json(json_path, slide, patientId)
        kfb2patientId.append([patientId, kfb_clsname, len(saved_img_info), kfb_path])
        total_saved_img_info.extend(saved_img_info)
        idx += 1
    
    df = pd.DataFrame(kfb2patientId, columns = ['patientId', 'kfb_clsname', 'valid_img_cnt', 'kfb_path'])
    df.to_csv(f'{save_dir}/JFSW_2_kfb_info.csv', index=False)
    
    df = pd.DataFrame(total_saved_img_info, columns = ['patientId', 'img_clsname', 'save_img_path'])
    df.to_csv(f'{save_dir}/JFSW_2_img_info.csv', index=False)

def filter_anno_kfb():
    keep_thr = 20
    df_JFSW_1 = pd.read_csv('data_resource/cls_pn/jfsw_valid_ann/JFSW_1_kfb_info.csv')
    df_JFSW_2 = pd.read_csv('data_resource/cls_pn/jfsw_valid_ann/JFSW_2_kfb_info.csv')
    merged_df = pd.concat([df_JFSW_1, df_JFSW_2], ignore_index=True)
    valid_img_cnts = []
    clasname_record = dict()
    keep_datas = []
    for row in merged_df.itertuples(index=False):
        cnt,clsname = row.valid_img_cnt, row.kfb_clsname
        valid_img_cnts.append(cnt)
        if cnt >= keep_thr:
            clasname_record[clsname] = clasname_record.get(clsname, 0) + 1
            keep_datas.append(row)
    keep_df = pd.DataFrame(keep_datas, columns=merged_df.columns)
    keep_df.to_csv('data_resource/cls_pn/jfsw_valid_ann/filtered_jfsw_anno.csv', index=False)

    bins = [0, 10, 20, 30, 40, 10000]
    axis_label = [f'{bins[i]}-{bins[i+1]}' for i in range(len(bins)-2)]
    axis_label.append('>40')
    counts, bin_edges = np.histogram(valid_img_cnts, bins=bins)
    
    result_table = PrettyTable(title='Valid Nums')
    result_table.field_names = ["区间"] + axis_label
    result_table.add_row(['nums'] + list(counts))
    print(result_table)

    result_table = PrettyTable(title=f'Valid Nums >= {keep_thr}')
    result_table.field_names = ["类别"] + list(clasname_record.keys())
    result_table.add_row(['nums'] + list(clasname_record.values()))
    print(result_table)


if __name__ == '__main__':
    # cut_valid_JFSW_1()
    # cut_valid_JFSW_2()
    filter_anno_kfb()
