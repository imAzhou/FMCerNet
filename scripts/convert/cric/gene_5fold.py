import json
import random
import os

def gene_abnormal():
    POSITIVE_CLASS = ['abnormal',]
    CLASS_COLORS = [[139,0,139]]
    WIDTH,HEIGHT = 1376,1020

    with open(f"{dataroot}/classifications.json", 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    raw_format = {}
    for item in raw_data:
        raw_format[item['image_id']] = item
    with open(f"{dataroot}/CRIC.json", 'r', encoding='utf-8') as f:
        DPD_data = json.load(f)
    
    for i in range(fold_num):
        train_data = random.sample(DPD_data, int(train_ratio*len(DPD_data)))
        train_ids = [item['imageId'] for item in train_data]
        val_data = [item for item in DPD_data if item['imageId'] not in train_ids]
        print(f'Fold-{i}: train ({len(train_data)}) val({len(val_data)})')
        
        for mode, datalist in zip(['train', 'val'],[train_data, val_data]):
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
            for idx,imgitem in enumerate(datalist):
                raw_item = raw_format[int(imgitem['imageId'])]
                image_name = raw_item['image_name']

                format_result['images'].append({
                    'id': idx, 
                    'width': WIDTH, 'height': HEIGHT,
                    'file_name': image_name
                })
                for child in imgitem['imageList']:
                    label = child['type']
                    x1,y1,xw,xh = child['x'],child['y'],child['w'],child['h']
                    x2,y2 = x1+xw,y1+xh
                    # 保证 bbox 不超出边界
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(WIDTH - 1, x2), min(HEIGHT - 1, y2)

                    format_result['annotations'].append({
                        "id": annid,
                        "image_id": idx,
                        "category_id": POSITIVE_CLASS.index(label) + 1,
                        "bbox": [x1,y1,xw,xh],
                        "area": xw*xh,
                        "iscrowd": 0,
                    })
                    annid += 1
            
            with open(f'{ann_savedir}/abnormal/flod{i}_{mode}.json', 'w', encoding='utf-8') as f:
                json.dump(format_result, f, ensure_ascii=False)


def gene_multicls():
    POSITIVE_CLASS = ['NILM', 'ASC-US','LSIL', 'ASC-H', 'HSIL', 'SCC']
    CLASS_COLORS = [[0,255,0], [31,119,180], [255,153,153], [255,105,180], [255,20,147], [139,0,139]]
    WIDTH,HEIGHT = 1376,1020

    with open(f"{dataroot}/classifications.json", 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    for i in range(fold_num):
        train_data = random.sample(raw_data, int(train_ratio*len(raw_data)))
        train_ids = [item['image_id'] for item in train_data]
        val_data = [item for item in raw_data if item['image_id'] not in train_ids]
        
        for mode, datalist in zip(['train', 'val'],[train_data, val_data]):
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
            for idx,imgitem in enumerate(datalist):
                format_result['images'].append({
                    'id': idx, 
                    'width': WIDTH, 'height': HEIGHT,
                    'file_name': imgitem['image_name']
                })

                for annitem in imgitem['classifications']:
                    label = annitem['bethesda_system']
                    if label == 'Negative for intraepithelial lesion':
                        label = 'NILM'
                    x,y = annitem['nucleus_x'],annitem['nucleus_y']
                    # 计算 bbox
                    box_size = 200
                    x1 = int(x - box_size / 2)
                    y1 = int(y - box_size / 2)
                    x2 = int(x + box_size / 2)
                    y2 = int(y + box_size / 2)

                    # 保证 bbox 不超出边界
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(WIDTH - 1, x2), min(HEIGHT - 1, y2)
                    format_result['annotations'].append({
                        "id": annid,
                        "image_id": idx,
                        "category_id": POSITIVE_CLASS.index(label) + 1,
                        "bbox": [x1,y1,box_size,box_size],
                        "area": box_size*box_size,
                        "iscrowd": 0,
                    })
                    annid += 1
            
            with open(f'{ann_savedir}/flod{i}_{mode}.json', 'w', encoding='utf-8') as f:
                json.dump(format_result, f, ensure_ascii=False)

if __name__ == "__main__":
    fold_num = 5
    train_ratio = 0.8
    dataroot = 'data_resource/CRIC'
    ann_savedir = f'{dataroot}/annofiles'
    os.makedirs(ann_savedir, exist_ok=True, mode=0o777)
    # gene_multicls()
    gene_abnormal()
