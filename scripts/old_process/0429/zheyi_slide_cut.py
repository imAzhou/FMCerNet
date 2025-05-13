import json
import os
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
import cv2
import torch
from mmpretrain.structures import DataSample
from cerwsi.utils import KFBSlide,draw_OD,is_bbox_inside,calc_relative_coord,generate_cut_regions,random_cut_square

CERTAIN_THR = 0.7
PATCH_EDGE = 224
STRIDE = 200
LEVEL = 0
downsample_ratio = 1
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
    from cerwsi.nets import ValidClsNet

    device = torch.device('cuda:0')
    valid_model_ckpt = 'checkpoints/valid_cls_best.pth'
    valid_model = ValidClsNet()
    valid_model.to(device)
    valid_model.eval()
    valid_model.load_state_dict(torch.load(valid_model_ckpt))

    pn_cnt = [0,0]
    for slideInfo in tqdm(total_slideInfo, ncols=80, desc='Cuting slide'):
        slide = KFBSlide(slideInfo['kfb_path'])
        for patchinfo in slideInfo['patchlist']:
            start_x, start_y = patchinfo['square_x1y1']
            location, level, size = (start_x, start_y), LEVEL, (PATCH_EDGE,PATCH_EDGE)
            read_result = Image.fromarray(slide.read_region(location, level, size))
            if patchinfo['diagnose'] == 1 or random.random() < keep_ratio:
                data_batch = dict(inputs=[], data_samples=[])
                img_input = cv2.cvtColor(np.array(read_result), cv2.COLOR_RGB2BGR)
                img_input = torch.as_tensor(cv2.resize(img_input, (224,224)))
                data_batch['inputs'].append(img_input.permute(2,0,1))    # (bs, 3, h, w)
                data_batch['data_samples'].append(DataSample())
                data_batch['inputs'] = torch.stack(data_batch['inputs'], dim=0)
                with torch.no_grad():
                    outputs = valid_model.val_step(data_batch)
                if max(outputs[0].pred_score) > CERTAIN_THR and outputs[0].pred_label == 1:
                    pn_cnt[patchinfo['diagnose']] += 1
                    save_dir = pos_save_dir if patchinfo['diagnose'] == 1 else neg_save_dir
                    read_result.save(f'{save_dir}/{patchinfo["filename"]}')
    print(pn_cnt)       

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
            cx1,cy1,cx2,cy2 = coord
            w,h = cx2-cx1, cy2-cy1
            if w >25 and h > 25:
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

def gene_ann_json(mode):
    df_data = pd.read_csv(f'data_resource/0429_2/annofiles/{mode}.csv')
    total_kfbInfo = []
    for clsname,slidelist in annotaion.items():
        for slideinfo in tqdm(slidelist, ncols=80, desc=f'Processing {clsname}'):
            imageName = slideinfo['imageName']
            patientId = name2PID[imageName]
            filtered_row = df_data.loc[df_data['patientId'] == patientId]
            if filtered_row.empty:
                continue
            kfb_path = filtered_row.iloc[0]['kfb_path']
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
                if sub_class in POSITIVE_CLASS and width>15 and height>15:
                    rect_items.append(dict(sub_class=sub_class, region=bbox_coords))
                    all_pos_bboxes.append(bbox_coords)
                if sub_class == 'RoI' and width>15 and height>15:
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

def imgName2patientId():
    name2PID = {}
    for row in tqdm(df_ann.itertuples(index=False), total=len(df_ann), ncols=80):
        filename = os.path.basename(row.kfb_path)
        name2PID[filename] = row.patientId
    return name2PID

if __name__ == '__main__':
    ann_json = 'data_resource/zheyi_annofiles/宫颈液基细胞—Slide-0422.json'
    with open(ann_json,'r') as f:
        annotaion = json.load(f)
    df_ann = pd.read_csv('data_resource/zheyi_annofiles/0422_slide_anno.csv')
    name2PID = imgName2patientId()
    for mode in ['train', 'val']:
        save_ann_json = f'data_resource/0429_2/annofiles/zheyi_slide_{mode}_{PATCH_EDGE}.json'
        # total_slideInfo = gene_ann_json(mode)
        # analyze_patchlist(total_slideInfo)
        # with open(save_ann_json,'w') as f:
        #     json.dump(total_slideInfo, f)
    
        keep_ratio = 0.5 if mode == 'val' else 0.05
        pos_save_dir = f'data_resource/0429_2/{PATCH_EDGE}/images/total_pos'
        os.makedirs(pos_save_dir, exist_ok=True)
        neg_save_dir = f'data_resource/0429_2/{PATCH_EDGE}/images/neg'
        os.makedirs(neg_save_dir, exist_ok=True)
        with open(save_ann_json,'r') as f:
            total_slideInfo = json.load(f)
        cut_patchlist(total_slideInfo)

'''
0409 (only train mode)
224
[84335, 3915] → [4637, 3387]
512
[16353, 2592] → [6823, 2490]

0422
224
train [121156, 5168] → [3312, 4990]
val [5873, 1187] → [2736, 1171]
512
train [23324, 3068] → [7355, 3055]
val [1009, 603] → [992, 603]
'''