from tqdm import tqdm
import pandas as pd
import uuid
import os
import json
from PIL import Image
import shutil
import glob
from cerwsi.utils import (KFBSlide,remap_points,read_json_anno,random_cut_square,
                          is_bbox_inside,calc_relative_coord)

POSITIVE_CLASS = ['ASC-US', 'LSIL', 'ASC-H', 'HSIL', 'SCC', 'AGC-NOS', 'AGC', 'AGC-N', 'AGC-FN']
RECORD_CLASS = {
    'ASC-US': 'ASC-US',
    'LSIL': 'LSIL',
    'ASC-H': 'ASC-H',
    'HSIL': 'HSIL',
    'SCC': 'HSIL',
    'AGC-N': 'AGC',
    'AGC': 'AGC',
    'AGC-NOS': 'AGC',
    'AGC-FN': 'AGC',
}

def get_ROI_inside(roi_rect, anns):
    roi_x,roi_y,roi_w,roi_h = roi_rect
    roi_x1y1x2y2 = [roi_x,roi_y,roi_x+roi_w,roi_y+roi_h]
    item_inside = []
    for ann_ in anns:
        ann = remap_points(ann_)
        if ann is None:
            continue
        region = ann.get('region')
        sub_class = ann.get('sub_class')
        x,y = region['x'],region['y']
        w,h = region['width'],region['height']
        if w>25 and h>25 and is_bbox_inside([x,y,x+w,y+h], roi_x1y1x2y2, tolerance=5) and sub_class in POSITIVE_CLASS:
            item_inside.append(ann)
    return item_inside

def formatAnno(ROI_inside, RoI_coords):
    result = []
    rx1,ry1,rx2,ry2 = RoI_coords
    for item in ROI_inside:
        time_id = int(str(uuid.uuid4().int)[:13])  # 取 UUID 转换成的整数的前 13 位
        region = item['region']
        x,y = region['x'],region['y']
        w,h = region['width'],region['height']
        target_bbox = [x,y,x+w,y+h]
        cx1,cy1,cx2,cy2 = calc_relative_coord([rx1,ry1,rx2,ry2], target_bbox)
        format_item = {
            'id': time_id,
            "type": "rect",
            "shape": "rect",
            "label": RECORD_CLASS[item["sub_class"]],
            "points": [
                [cx1,cy1],
                [cx2,cy1],
                [cx2,cy2],
                [cx1,cy2]]}
        result.append(format_item)
    return result

def gene_patch_json(train_csv_file, val_csv_file):
    train_data_df = pd.read_csv(train_csv_file)
    val_data_df = pd.read_csv(val_csv_file)

    cnt = 0
    for data_df,mode in zip([train_data_df,val_data_df], ['train','val']):

        for row in tqdm(data_df.itertuples(index=True), total=len(data_df)):
            if row.kfb_source == 'WXL_1' or row.kfb_clsname in ['NILM', 'ASC-US', 'AGC']:
                continue

            filename = f'{row.patientId}_RoI'
            if filename in exist_list:
                continue
            json_path = f'/medical-data/data/{row.anno_path}'
            annos = read_json_anno(json_path)
            inside_cnt = []
            for annidx, ann_ in enumerate(annos):
                ann = remap_points(ann_)
                if ann is None:
                    continue
                sub_class = ann.get('sub_class')
                region = ann.get('region')
                x,y = region['x'],region['y']
                w,h = region['width'],region['height']

                minv,maxv = 2000,5000
                if (minv <= w <= maxv) and (minv <= h <= maxv) and sub_class == 'ROI':
                    RoI_coords = [x,y,w,h]
                    ROI_inside = get_ROI_inside(RoI_coords, annos)
                    inside_cnt.append({
                        'ann': ann,
                        'ROI_inside': ROI_inside,
                        'inside_num': len(ROI_inside),
                    })
            inside_cnt.sort(key=lambda x: x['inside_num'], reverse=True)
            slide = KFBSlide(row.kfb_path)
            swidth, sheight = slide.level_dimensions[0]
            annotationResult = []
            
            if len(inside_cnt) > 0:
                RoI_region = inside_cnt[0]['ann']
                region = RoI_region.get('region')
                x,y = region['x'],region['y']
                w,h = region['width'],region['height']
                annotationResult = formatAnno(inside_cnt[0]['ROI_inside'], [x,y,x+w,y+h])
                location, level, size = (x,y), 0, (w,h)
                read_result = Image.fromarray(slide.read_region(location, level, size))
                read_result.save(f'data_resource/0328/RoI_label/group5/img/{filename}.png')
                with open(f'data_resource/0328/RoI_label/group5/ann/{filename}.json', 'w') as f:
                    json.dump({
                        "imageName": f"{filename}.png",
                        "annotationResult": annotationResult
                    }, f)
                cnt += 1
            
            # else:
            #     os.remove(f'{img_save_dir}/{filename}.png')
            #     os.remove(f'{ann_save_dir}/{filename}.json')
                # x,y = random_cut_square((swidth//4,sheight//4,swidth//2,sheight//2), maxv)
                # w = h = maxv

                # empty_cnt += 1

def random_cut(train_csv_file, val_csv_file):
    train_data_df = pd.read_csv(train_csv_file)
    val_data_df = pd.read_csv(val_csv_file)

    cnt = 0
    for data_df,mode in zip([train_data_df,val_data_df], ['train','val']):

        for row in tqdm(data_df.itertuples(index=True), total=len(data_df)):
            if row.kfb_source == 'WXL_1' or row.kfb_clsname in ['NILM', 'ASC-US', 'AGC']:
                continue

            filename = f'{row.patientId}_RoI'
            if filename in exist_list:
                continue
            slide = KFBSlide(row.kfb_path)
            swidth, sheight = slide.level_dimensions[0]
            
            maxv = 4000
            x,y = random_cut_square((swidth//4,sheight//4,swidth//2,sheight//2), maxv)
            w = h = maxv
            location, level, size = (x,y), 0, (w,h)
            read_result = Image.fromarray(slide.read_region(location, level, size))
            read_result.save(f'data_resource/0328/RoI_label/group6/img/{filename}.png')
            with open(f'data_resource/0328/RoI_label/group6/ann/{filename}.json', 'w') as f:
                json.dump({
                    "imageName": f"{filename}.png",
                    "annotationResult": []
                }, f)
            scale = 0.1
            new_size = (int(read_result.width * scale), int(read_result.height * scale))
            img_resized = read_result.resize(new_size, Image.LANCZOS)
            img_resized.save(f'thumbnile/{filename}.png')
            cnt += 1
    print(cnt)

if __name__ == '__main__':
    train_csv_file = 'data_resource/slide_anno/0319/train.csv'
    val_csv_file = 'data_resource/slide_anno/0319/val.csv'
    save_dir = 'data_resource/0328/RoI_label'
    
    exist_list = []
    for i in [1,2,3,4,5]:
        all_imgs = os.listdir(f'data_resource/0328/RoI_label/group{i}/img')
        exist_list.extend([name.split('.')[0] for name in all_imgs])
    # gene_patch_json(train_csv_file, val_csv_file)
    random_cut(train_csv_file, val_csv_file)
