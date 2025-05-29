'''
存在被包含的bbox以及非常小的bbox，宽高只有几个像素
原论文没有处理以上情况，为保持一致我们也留下这些 GT bbox
有一张图片由于没有有效标注框，被源码过滤了：2116336_0079.png
'''
import uuid
import json
import os
import pandas as pd
from tqdm import tqdm
from PIL import Image


def parse_annos(annotations, parentId):
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
            annid = int(str(uuid.uuid4().int)[:13]),
            sub_class='abnormal', region=[xmin, ymin, xmax, ymax],
            parent_id = parentId
        ))
    
    return rect_items


def gene_filter():
    slide_filter_items = []
    for df_data in [df_train, df_val, df_test]:
        for row in tqdm(df_data.itertuples(index=False), total=len(df_data), ncols=80):
            imgname = os.path.basename(row.image_path)
            imgpath = f'{data_root}/JPEGImages/{imgname}'
            img = Image.open(imgpath)
            w,h = img.size
            roiAnnID = imgname.split('.')[0].split('_')[1]
            rect_items = parse_annos(row.annotation, roiAnnID)
            roiItem = dict(
                annid = roiAnnID,   # str
                sub_class = 'RoI',
                region = [0,0,w,h],
                parent_id = -1,
                children = rect_items
            )
            slide_filter_items.append({
                'patientId': row.patient_id,
                'media_type': 'roi',
                'source_path': imgpath,
                'annotations': [roiItem]
            })
    with open(f'{ann_savedir}/unify_ann.json', 'w', encoding='utf-8') as f:
        json.dump(slide_filter_items, f, ensure_ascii=False)



if __name__ == "__main__":
    data_root = '/medical-data/data/cervix/HMCHH'
    ann_savedir = f'data_resource/HMCHH/annofiles'
    os.makedirs(ann_savedir, exist_ok=True, mode=0o777)
    df_train = pd.read_csv('data_resource/HMCHH/csvfiles/fold1/train.csv')
    df_val = pd.read_csv('data_resource/HMCHH/csvfiles/fold1/val.csv')
    df_test = pd.read_csv('data_resource/HMCHH/csvfiles/test.csv')

    gene_filter()
