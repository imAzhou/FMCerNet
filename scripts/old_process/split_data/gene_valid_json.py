import pandas as pd
import json
import os
from tqdm import tqdm
from pathlib import Path
from cerwsi.utils import KFBSlide, read_json_valid
import xml.etree.ElementTree as ET

def process_wxl_1(row_info):
    kfb_path,kfb_clsname = row_info.kfb_path, row_info.kfb_clsname
    valid_info = dict(
        kfb_path = kfb_path,
        kfb_clsname = kfb_clsname,
        patientId = row_info.patientId,
        valid_num = 0,
        valid_anno = []
    )
    xml_path = kfb_path.replace('.kfb','.xml')
    slide = KFBSlide(kfb_path)
    max_x, max_y = slide.level_dimensions[0]
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for region in root.findall('.//Region'):
        coords = []
        for vertex in region.findall('.//Vertex'):
            # 获取 Vertex 节点的 X 和 Y 属性值
            x = vertex.get('X')
            y = vertex.get('Y')
            coords.append((x,y))
        (start_x,start_y),(end_x, end_y) = coords[0],coords[2]
        start_x,start_y,end_x,end_y = round(float(start_x)), round(float(start_y)), round(float(end_x)), round(float(end_y))
        x1,y1 = min(start_x,end_x),min(start_y,end_y)
        x2,y2 = max(start_x,end_x),max(start_y,end_y)
        x2,y2 = min(x2,max_x), min(y2,max_y)

        w, h = x2 - x1, y2 - y1
        if w > 32 and h > 32:
            valid_info['valid_anno'].append(dict(
                coord = (x1,y1,x2,y2),
                size = (w,h),
                patch_clsname = kfb_clsname
            ))
    valid_info['valid_num'] = len(valid_info['valid_anno'])

    return valid_info

def process_jfsw(row_info, json_path):
    kfb_path,kfb_clsname = row_info.kfb_path, row_info.kfb_clsname
    valid_info = dict(
        kfb_path = kfb_path,
        kfb_clsname = kfb_clsname,
        patientId = row_info.patientId,
        valid_num = 0,
        valid_anno = []
    )
    slide = KFBSlide(kfb_path)
    max_x, max_y = slide.level_dimensions[0]
    valid_annos = read_json_valid(json_path, (max_x, max_y))
    valid_info['valid_anno'] = valid_annos
    valid_info['valid_num'] = len(valid_annos)

    return valid_info

if __name__ == '__main__':
    df_train = pd.read_csv('data_resource/cls_pn/1127_train.csv')
    df_val = pd.read_csv('data_resource/cls_pn/1127_val.csv')

    anno_train = dict(slide_num = df_train.shape[0], valid_imgs = [])
    anno_val = dict(slide_num = df_val.shape[0], valid_imgs = [])

    for df_data, anno_json in zip([df_train, df_val], [anno_train, anno_val]):
        for row in tqdm(df_data.itertuples(index=False), total=len(df_data)):
            slide_valid_info = None
            if row.kfb_source == 'WXL_1' and row.kfb_clsid == 1:
                slide_valid_info = process_wxl_1(row)
            elif row.kfb_source == 'JFSW_1' and row.kfb_clsid == 1:
                filename = os.path.basename(row.kfb_path)
                json_path = f'/disk/medical_datasets/cervix/JFSW/阳性json/{filename}'.replace('.kfb','.json')
                slide_valid_info = process_jfsw(row, json_path)
            elif row.kfb_source == 'JFSW_2':
                path = Path(row.kfb_path)
                directories = path.parents
                filename = os.path.basename(row.kfb_path).replace('.kfb','.json')
                json_path = f'{directories[0]}/json/{filename}'
                slide_valid_info = process_jfsw(row, json_path)
            
            if slide_valid_info is not None:
                anno_json['valid_imgs'].append(slide_valid_info)
    
    with open('data_resource/cls_pn/1127_anno_train.json', 'w') as file:
        json.dump(anno_train, file)
    with open('data_resource/cls_pn/1127_anno_val.json', 'w') as file:
        json.dump(anno_val, file)
