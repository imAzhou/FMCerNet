import json
import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from cerwsi.utils import draw_OD,is_bbox_inside,calc_relative_coord,generate_cut_regions,random_cut_square

PATCH_EDGE = 512
STRIDE = 450
POSITIVE_CLASS = ['AGC', 'ASC-US','LSIL', 'ASC-H', 'HSIL']
magnify = '20x' # 40x
downsample_ratio = 1 if magnify == '40x' else 2

invalid_pids = [
    'JFSW_1_57', 'JSFW_1_67', 'JSFW_2_125', 'JSFW_2_1323', 'JSFW_2_1327', 'JSFW_2_1583', 'JSFW_2_1521',
    'JFSW_2_837',
    'JFSW_2_1308', 'JFSW_2_255', 'JFSW_2_360',
]

def analyze_patchlist(total_roiInfo):
    total_patchlist = []
    for item in total_roiInfo:
        total_patchlist.extend(item['patchlist'])
    
    pn_cnt = [0,0]
    for patchinfo in total_patchlist:
        pn_cnt[patchinfo['diagnose']] += 1
    print(pn_cnt)
        
def cut_patchlist(total_roiInfo):
    pos_save_dir = 'data_resource/0416/images/total_pos'
    os.makedirs(pos_save_dir, exist_ok=True)
    neg_save_dir = 'data_resource/0416/images/neg'
    os.makedirs(neg_save_dir, exist_ok=True)
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
            cropped = img.crop((sx1,sy1,sx2,sy2))
            save_dir = pos_save_dir if patchinfo['diagnose'] == 1 else neg_save_dir
            save_path = f'{save_dir}/{patchinfo["filename"]}'
            cropped.save(save_path)

def gene_ann_json():
    train_data_df = pd.read_csv('data_resource/0416/annofiles/train.csv')
    val_data_df = pd.read_csv('data_resource/0416/annofiles/val.csv')
    ann_json = 'data_resource/0328/annofiles/宫颈液基细胞—RoI.json'
    with open(ann_json,'r') as f:
        annotaion = json.load(f)

    Group1 = annotaion['Group1']
    Group3 = annotaion['Group3']
    Group4 = annotaion['Group4']

    # vis_patientIds = ['JFSW_1_9','JFSW_2_1491','JFSW_2_688','JFSW_2_1353','JFSW_2_692']
    vis_patientIds = ['JFSW_2_548']
    train_roiInfo,val_roiInfo = [],[]

    for itemlist,tag in zip([Group1,Group3,Group4],['group1','group3','group4']):
        for imgitem in tqdm(itemlist, ncols=80):
            filename = imgitem['imageName']
            patientId = filename.replace('_RoI.png','')
            # if patientId not in vis_patientIds:
            #     continue
            if patientId in invalid_pids:
                continue

            img_path = f'data_resource/0328/{tag}/img/{filename}'
            img = Image.open(img_path)
            imgw,imgh = img.size
            imgw /= downsample_ratio
            imgh /= downsample_ratio
            imgw,imgh = round(imgw),round(imgh)
            
            # dict(sub_class:str,region:[ x1, y1, x2, y2]) JFSW_2_1565_RoI
            annos = imgitem['annotations'][0]['annotationResult']
            rect_items = []
            for annitem in annos:
                sub_class = annitem['label']
                if annitem['type'] != 'circle' and sub_class in POSITIVE_CLASS:
                    all_x,all_y = [p[0] for p in annitem['points']],[p[1] for p in annitem['points']]
                    x1,x2 = min(all_x),max(all_x)
                    y1,y2 = min(all_y),max(all_y)
                    rect_items.append(dict(sub_class=sub_class,
                                           region=(np.array([x1, y1, x2, y2]) // downsample_ratio).tolist()))
            
            roiInfo = {
                'patientId': patientId,
                'roi_imgpath': img_path,
                'source': f'zheyi_roi_{tag}',
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
                for rect in rect_items:
                    sub_class = rect['sub_class']
                    target_bbox = rect['region']
                    coord = calc_relative_coord([sx1,sy1,sx2,sy2],target_bbox)
                    if coord is not None:
                        patchinfo['bboxes'].append(coord)
                        patchinfo['clsnames'].append(sub_class)
                        patchinfo['diagnose'] = 1
                        patchinfo['prefix'] = 'total_pos'

                if patchinfo['diagnose'] == 1:
                    roiInfo['patchlist'].append(patchinfo)
                elif patchinfo['diagnose'] == 0 and random.random() < 0.5:
                    roiInfo['patchlist'].append(patchinfo)
            
            roiInfo['patchlist'] = filter_patch_list(roiInfo['patchlist'])
            
            # 判断是否有值为 'value' 的行
            if (train_data_df['patientId'] == patientId).any():
                train_roiInfo.append(roiInfo)
            else:
                val_roiInfo.append(roiInfo)

    return train_roiInfo,val_roiInfo

def filter_patch_list(roi_patch_list):
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

    new_patch_list = []
    for patchinfo in roi_patch_list:
        if len(patchinfo['bboxes']) != 1:
            new_bboxes,nes_clsnames = [],[]
            for bbox,clsname in zip(patchinfo['bboxes'], patchinfo['clsnames']):
                if pos_box_enclosured(bbox, patchinfo['bboxes']):
                    continue
                new_bboxes.append(bbox)
                nes_clsnames.append(clsname)
            patchinfo['bboxes'] = new_bboxes
            patchinfo['clsnames'] = nes_clsnames
            
        if len(patchinfo['bboxes']) == 0 and patchinfo['diagnose'] != 0:
            continue
        new_patch_list.append(patchinfo)

    return new_patch_list



if __name__ == '__main__':
    
    ann_save_dir = 'data_resource/0416/annofiles'
    img_save_dir = 'data_resource/0416/images/total_pos'
    os.makedirs(img_save_dir, exist_ok=True)
    
    # train_roiInfo,val_roiInfo = gene_ann_json()
    # for mode,roiInfo in zip(['train','val'],[train_roiInfo,val_roiInfo]):
    #     analyze_patchlist(roiInfo)
    #     with open(f'{ann_save_dir}/{mode}_roi_total.json','w') as f:
    #         json.dump(roiInfo, f)
    
    for mode in ['val']:
        save_ann_json = f'{ann_save_dir}/{mode}_roi_total.json'
        with open(save_ann_json,'r') as f:
            total_roiInfo = json.load(f)
        cut_patchlist(total_roiInfo)


'''
train
Neg,Pos: [8836, 6912]

val
Neg,Pos: [2351, 1818]
'''