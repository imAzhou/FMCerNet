import json
import os
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import cv2
from cerwsi.nets import ValidClsNet
from mmpretrain.structures import DataSample
from cerwsi.utils import draw_OD,is_bbox_inside,calc_relative_coord,generate_cut_regions,random_cut_square

CERTAIN_THR = 0.7
PATCH_EDGE = 512
STRIDE = 450
POSITIVE_CLASS = ['AGC', 'ASC-US','LSIL', 'ASC-H', 'HSIL']
magnify = '40x' # 40x
downsample_ratio = 1 if magnify == '40x' else 2

def analyze_patchlist(total_roiInfo):
    total_patchlist = []
    for item in total_roiInfo:
        total_patchlist.extend(item['patchlist'])
    
    pn_cnt = [0,0]
    for patchinfo in total_patchlist:
        pn_cnt[patchinfo['diagnose']] += 1
    print(pn_cnt)
        
def cut_patchlist(total_roiInfo):
    device = torch.device('cuda:0')
    valid_model_ckpt = 'checkpoints/valid_cls_best.pth'
    valid_model = ValidClsNet()
    valid_model.to(device)
    valid_model.eval()
    valid_model.load_state_dict(torch.load(valid_model_ckpt))

    pn_cnt = [0,0]
    for roiInfo in tqdm(total_roiInfo, ncols=80, desc='Cuting img'):
        img = Image.open(roiInfo['roi_imgpath'])
        imgw,imgh = img.size
        imgw /= downsample_ratio
        imgh /= downsample_ratio
        imgw,imgh = round(imgw),round(imgh)
        if downsample_ratio !=1:
            img = img.resize((imgw,imgh))

        for patchinfo in roiInfo['patchlist']:
            start_x, start_y = patchinfo['square_x1y1']
            sx1,sy1,sx2,sy2 = start_x, start_y, start_x+PATCH_EDGE, start_y+PATCH_EDGE
            # if patchinfo['filename'] != 'JFSW_1_9_roi187.png':
            #     continue
            cropped = img.crop((sx1,sy1,sx2,sy2))
            if patchinfo['diagnose'] == 1 or random.random() < 0.1:
                data_batch = dict(inputs=[], data_samples=[])
                img_input = cv2.cvtColor(np.array(cropped), cv2.COLOR_RGB2BGR)
                img_input = torch.as_tensor(cv2.resize(img_input, (224,224)))
                data_batch['inputs'].append(img_input.permute(2,0,1))    # (bs, 3, h, w)
                data_batch['data_samples'].append(DataSample())
                data_batch['inputs'] = torch.stack(data_batch['inputs'], dim=0)
                with torch.no_grad():
                    outputs = valid_model.val_step(data_batch)
                if max(outputs[0].pred_score) > CERTAIN_THR and outputs[0].pred_label == 1:
                    pn_cnt[patchinfo['diagnose']] += 1
                    save_dir = pos_save_dir if patchinfo['diagnose'] == 1 else neg_save_dir
                    cropped.save(f'{save_dir}/{patchinfo["filename"]}')
    print(pn_cnt)

def gene_ann_json():
    train_data_df = pd.read_csv('data_resource/0416/annofiles/train.csv')
    val_data_df = pd.read_csv('data_resource/0416/annofiles/val.csv')
    ann_json = 'data_resource/0328/annofiles/宫颈液基细胞—RoI_filter.json'
    with open(ann_json,'r') as f:
        annotaion = json.load(f)

    train_roiInfo,val_roiInfo = [],[]
    for imgitem in tqdm(annotaion, ncols=80):
        filename = imgitem['imageName']
        patientId = filename.replace('_RoI.png','')
        # if patientId not in vis_patientIds:
        #     continue

        img_path = f'data_resource/0328/{imgitem["tag"]}/img/{filename}'
        img = Image.open(img_path)
        imgw,imgh = img.size
        imgw /= downsample_ratio
        imgh /= downsample_ratio
        imgw,imgh = round(imgw),round(imgh)
        
        rect_items = []
        for annitem in imgitem['annotations']:
            sub_class = annitem['label']
            if annitem['type'] != 'circle' and sub_class in POSITIVE_CLASS:
                all_x,all_y = [p[0] for p in annitem['points']],[p[1] for p in annitem['points']]
                x1,x2 = min(all_x),max(all_x)
                y1,y2 = min(all_y),max(all_y)
                rect_items.append(dict(sub_class=sub_class,
                                        region=(np.array([x1, y1, x2, y2]) // downsample_ratio).tolist()))
        no_inside_rect_items = filter_anno_list(rect_items)
        roiInfo = {
            'patientId': patientId,
            'roi_imgpath': img_path,
            'source': f'zheyi_roi_{imgitem["tag"]}',
            'patchlist': []
        }

        cut_coords = generate_cut_regions((0,0), imgw, imgh, PATCH_EDGE, STRIDE)
        random_coords = [random_cut_square((0,0,imgw,imgh),PATCH_EDGE) for i in range(10)]
        for idx, (start_x, start_y) in enumerate([*cut_coords, *random_coords]):
            patchinfo = {
                'filename':f'{patientId}_roi{idx}.png',
                'square_x1y1': (start_x, start_y),
                'bboxes': [],
                'clsnames': [],
                'prefix': 'neg',
                'diagnose': 0
            }
            sx1,sy1,sx2,sy2 = start_x, start_y, start_x+PATCH_EDGE, start_y+PATCH_EDGE
            for rect in no_inside_rect_items:
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
                roiInfo['patchlist'].append(patchinfo)
            elif patchinfo['diagnose'] == 0 and random.random() < 0.1:
                roiInfo['patchlist'].append(patchinfo)
        
        # 判断是否有值为 'value' 的行
        if (train_data_df['patientId'] == patientId).any():
            train_roiInfo.append(roiInfo)
        else:
            val_roiInfo.append(roiInfo)

    return train_roiInfo,val_roiInfo

def filter_anno_list(rect_items):
    '''
    去掉阳性框内部的小框
    '''

    def pos_box_enclosured(bbox, all_bboxes):
        for parent_bbox in all_bboxes:
            if bbox[0] == parent_bbox[0] and bbox[1] == parent_bbox[1] and bbox[2] == parent_bbox[2] and bbox[3] == parent_bbox[3]:
                continue
            if is_bbox_inside(bbox, parent_bbox, 5):
                return True
        return False

    if len(rect_items) == 1:
        return rect_items
    
    new_rect_items = []
    total_bboxes = [i['region'] for i in rect_items]
    for rect in rect_items:
        target_bbox = rect['region']
        if pos_box_enclosured(target_bbox, total_bboxes):
            continue
        new_rect_items.append(rect)

    return new_rect_items


if __name__ == '__main__':
    
    ann_save_dir = 'data_resource/0429/annofiles'
    os.makedirs(ann_save_dir, exist_ok=True)
    
    # train_roiInfo,val_roiInfo = gene_ann_json()
    # for mode,roiInfo in zip(['train','val'],[train_roiInfo,val_roiInfo]):
    #     analyze_patchlist(roiInfo)
    #     with open(f'{ann_save_dir}/roi_{mode}_512.json','w') as f:
    #         json.dump(roiInfo, f)
    
    pos_save_dir = 'data_resource/0429/512/images/total_pos'
    os.makedirs(pos_save_dir, exist_ok=True)
    neg_save_dir = 'data_resource/0429/512/images/neg'
    os.makedirs(neg_save_dir, exist_ok=True)
    for mode in ['train','val']:
        save_ann_json = f'{ann_save_dir}/roi_{mode}_512.json'
        with open(save_ann_json,'r') as f:
            total_roiInfo = json.load(f)
        cut_patchlist(total_roiInfo)


'''
224 200
train Neg,Pos: [12837, 10595]
val Neg,Pos: [3375, 3140]

512 450
train Neg,Pos: [2777, 8121]
val Neg,Pos: [753, 2324]
'''