import json
import os
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
from cerwsi.utils import KFBSlide,draw_OD,is_bbox_inside,calc_relative_coord,generate_cut_regions,random_cut_square

PATCH_EDGE = 512
STRIDE = 450
LEVEL = 1
downsample_ratio = 2
POSITIVE_CLASS = ['AGC', 'ASC-US','LSIL', 'ASC-H', 'HSIL']

def analyze_patchlist(total_roiInfo):
    total_patchlist = []
    for item in total_roiInfo:
        total_patchlist.extend(item['patchlist'])
    
    pn_cnt = [0,0]
    for patchinfo in total_patchlist:
        pn_cnt[patchinfo['diagnose']] += 1
    print(pn_cnt)
        
def cut_patchlist(total_slideInfo):
    pos_save_dir = 'data_resource/0416/images/total_pos'
    os.makedirs(pos_save_dir, exist_ok=True)
    neg_save_dir = 'data_resource/0416/images/neg'
    os.makedirs(neg_save_dir, exist_ok=True)
    for slideInfo in tqdm(total_slideInfo, ncols=80, desc='Cuting slide'):
        slide = KFBSlide(slideInfo['kfb_path'])
        for patchinfo in slideInfo['patchlist']:
            start_x, start_y = patchinfo['square_x1y1']
            location, level, size = (start_x, start_y), LEVEL, (PATCH_EDGE,PATCH_EDGE)
            read_result = Image.fromarray(slide.read_region(location, level, size))
            save_dir = pos_save_dir if patchinfo['diagnose'] == 1 else neg_save_dir
            save_path = f'{save_dir}/{patchinfo["filename"]}'
            read_result.save(save_path)

def gene_patchinfo(patch_coords,patch_filename,pos_rect_items):
    sx1,sy1,sx2,sy2 = patch_coords
    patchinfo = {
        'filename':patch_filename,
        'square_x1y1': (sx1,sy1),
        'bboxes': [],
        'clsnames': [],
        'prefix': 'neg',
        'diagnose': 0
    }
    for rect in pos_rect_items:
        sub_class = rect['sub_class']
        target_bbox = rect['region']
        coord = calc_relative_coord([sx1,sy1,sx2,sy2],target_bbox)
        if coord is not None:
            patchinfo['bboxes'].append(coord)
            patchinfo['clsnames'].append(sub_class)
            patchinfo['diagnose'] = 1
            patchinfo['prefix'] = 'total_pos'
    if patchinfo['diagnose'] == 1:
        return patchinfo
    elif patchinfo['diagnose'] == 0 and random.random() < 0.5:
        return patchinfo
    
    return None

def gene_RoI_patchlist(patientId, roi_items, pos_rect_items):
    patchlist = []
    for ridx,(rx1,ry1,rx2,ry2) in enumerate(roi_items):
        width = rx2 - rx1
        height = ry2 - ry1
        cut_coords = generate_cut_regions((rx1,ry1), width, height, PATCH_EDGE, STRIDE)
        random_coords = [random_cut_square((rx1,ry1,width,height),PATCH_EDGE) for i in range(10)]
        for idx, (start_x, start_y) in enumerate([*cut_coords, *random_coords]):
            patch_coords = [start_x, start_y, start_x+PATCH_EDGE, start_y+PATCH_EDGE]
            patchinfo = gene_patchinfo(patch_coords,f'{patientId}_roi{ridx}_{idx}.png',pos_rect_items)
            if patchinfo is not None:
                patchlist.append(patchinfo)
    return patchlist

def gene_WSI_patchlist(patientId,pos_rect_items):
    patchlist = []
    for idx,rect in enumerate(pos_rect_items):
        target_bbox = rect['region']    # (x1,y1,x2,y2)
        tx1,ty1,tx2,ty2 = target_bbox
        width,height = tx2-tx1, ty2-ty1
        start_x, start_y = random_cut_square((tx1,ty1,width,height),PATCH_EDGE)
        patch_coords = [start_x, start_y, start_x+PATCH_EDGE, start_y+PATCH_EDGE]
        patchinfo = gene_patchinfo(patch_coords,f'{patientId}_{idx}.png',pos_rect_items)
        if patchinfo is not None:
            patchlist.append(patchinfo)
    return patchlist

def pos_box_enclosured(bbox, all_bboxes):
    for parent_bbox in all_bboxes:
        if bbox[0] == parent_bbox[0] and bbox[1] == parent_bbox[1] and bbox[2] == parent_bbox[2] and bbox[3] == parent_bbox[3]:
            continue
        if is_bbox_inside(bbox, parent_bbox, 5):
            return True
    return False

def gene_ann_json():
    ann_json = 'data_resource/0328/annofiles/宫颈液基细胞—Slide.json'
    kfb_root_dir = '/nfs-medical/vipa-medical/zheyi/zly/KFBs/till_0318'
    with open(ann_json,'r') as f:
        annotaion = json.load(f)
    
    train_csv_file = 'data_resource/0416/annofiles/train.csv'
    df_train = pd.read_csv(train_csv_file)
    
    total_kfbInfo = []
    for clsname,slidelist in annotaion.items():
        for slideinfo in tqdm(slidelist, ncols=80, desc=f'Processing {clsname}'):
            imageName = slideinfo['imageName']
            kfb_path = f'{kfb_root_dir}/{clsname}/{imageName}'
            filtered = df_train.loc[df_train['kfb_path'] == kfb_path].iloc[0]
            patientId = filtered['patientId']
            slide_anno = slideinfo['annotations'][0]['annotationResult']
            all_clsname = [annitem['label'] for annitem in slide_anno]
            rect_items, all_pos_bboxes = [],[]
            roi_items = []
            for annitem in slide_anno:
                sub_class = annitem['label']
                all_x,all_y = [p[0] for p in annitem['points']],[p[1] for p in annitem['points']]
                x1,x2 = min(all_x),max(all_x)
                y1,y2 = min(all_y),max(all_y)
                bbox_coords = (np.array([x1, y1, x2, y2]) // downsample_ratio).tolist()
                width,height = x2-x1,y2-y1
                if sub_class in POSITIVE_CLASS and width>5 and height>5:
                    rect_items.append(dict(sub_class=sub_class, region=bbox_coords))
                    all_pos_bboxes.append(bbox_coords)
                if sub_class == 'RoI' and width>5 and height>5:
                    roi_items.append(bbox_coords)
            
            new_rect_items = []
            for ritem in rect_items:
                if pos_box_enclosured(ritem['region'], all_pos_bboxes):
                    continue
                new_rect_items.append(ritem)

            if 'RoI' in all_clsname:
                patchlist = gene_RoI_patchlist(patientId,roi_items,new_rect_items)
            else:
                patchlist = gene_WSI_patchlist(patientId,new_rect_items)
            kfbInfo = {
                'patientId': patientId,
                'kfb_path': kfb_path,
                'patchlist': patchlist
            }
            total_kfbInfo.append(kfbInfo)

    return total_kfbInfo


if __name__ == '__main__':
    
    save_ann_json = 'data_resource/0416/annofiles/train_slide_total.json'

    # total_slideInfo = gene_ann_json()
    # analyze_patchlist(total_slideInfo)
    # with open(save_ann_json,'w') as f:
    #     json.dump(total_slideInfo, f)
    
    with open(save_ann_json,'r') as f:
        total_slideInfo = json.load(f)
    cut_patchlist(total_slideInfo)
    # analyze_patchlist(total_slideInfo)

'''
Neg,Pos
[3824, 1662]
'''