
import json
import pandas as pd
from collections import defaultdict

from tqdm import tqdm

PATCH_EDGE = 512
POSITIVE_CLASS = ['AGC', 'ASC-US','LSIL', 'ASC-H', 'HSIL']
CLASS_COLORS = [[31,119,180], [255,153,153], [255,105,180], [255,20,147], [139,0,139]]

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
            {'id': idx, 'file_name': pInfo['filename'], 'width': PATCH_EDGE, 'height': PATCH_EDGE,
             'prefix': pInfo['prefix'], 'diagnose': pInfo['diagnose'], 'maskfile': pInfo['maskfile']})

        for bbox,clsname in zip(pInfo['bboxes'],pInfo['clsnames']):
            x1,y1,x2,y2 = bbox
            w,h = x2-x1, y2-y1
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


def main():
    with open('data_resource/0511/ann_jsons/patches_in_NegSlide.json', 'r', encoding='utf-8') as f:
        negslide_patchlist = json.load(f)
    with open('data_resource/0511/ann_jsons/patches_in_RoI_valid.json', 'r', encoding='utf-8') as f:
        RoI_patchlist = json.load(f)
    
    patient2patchlist = defaultdict(list)
    for patchInfo in [*negslide_patchlist, *RoI_patchlist]:
        patient2patchlist[patchInfo['patientId']].append(patchInfo)
    
    data_group = {
        'puretrain': 'data_resource/0511/4_pure_train.csv',
        'fusiontrain': 'data_resource/0511/5_fusion_train.csv',
        'val': 'data_resource/0511/6_val.csv'
    }
    for tag,csvpath in data_group.items():
        df_data = pd.read_csv(csvpath)
        patchlist = []
        for row in tqdm(df_data.itertuples(index=False), total=len(df_data), ncols=80):
            patchlist.extend(patient2patchlist[row.patientId])
        patchInCOCO = coco_format(patchlist)
        with open(f'{tag}_coco.json', 'w', encoding='utf-8') as f:
            json.dump(patchInCOCO, f, ensure_ascii=False)


if __name__ == "__main__":
    main()