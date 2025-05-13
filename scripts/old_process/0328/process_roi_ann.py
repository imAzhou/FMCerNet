import json
import os
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
from cerwsi.utils import draw_OD,is_bbox_inside,calc_relative_coord,generate_cut_regions,random_cut_square

PATCH_EDGE = 700
STRIDE = 650
POSITIVE_CLASS = ['AGC', 'ASC-US','LSIL', 'ASC-H', 'HSIL']
magnify = '20x' # 40x
downsample_ratio = 1 if magnify == '40x' else 2

invalid_pids = [
    'JFSW_1_57', 'JFSW_1_67', 'JFSW_2_125', 'JFSW_2_1323', 'JFSW_2_1327', 'JFSW_2_1583', 'JFSW_2_1521',
    'JFSW_2_837',
    'JFSW_2_1308', 'JFSW_2_255', 'JFSW_2_360', 'JFSW_2_278'
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
    pos_save_dir = 'data_resource/0328/images4fusion/Pos'
    os.makedirs(pos_save_dir, exist_ok=True)
    neg_save_dir = 'data_resource/0328/images4fusion/Neg'
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
    ann_json = 'data_resource/0328/annofiles/宫颈液基细胞—RoI.json'
    with open(ann_json,'r') as f:
        annotaion = json.load(f)
    anno_vis_dir = 'statistic_results/0328/zheyi_roi/'
    Group1 = annotaion['Group1']
    Group3 = annotaion['Group3']
    Group4 = annotaion['Group4']

    # vis_patientIds = ['JFSW_1_9','JFSW_2_1491','JFSW_2_688','JFSW_2_1353','JFSW_2_692']
    vis_patientIds = ['JFSW_2_548']
    total_roiInfo = []
    ann_type = [0,0,0]  # ['circle', 'rect', 'polygon']
    ann_clsnames = []
    beside_clsname_cnt = 0
    for itemlist,tag in zip([Group1,Group3,Group4],['group1','group3','group4']):
        group_anno_vis_dir = anno_vis_dir + tag + '_cropped'
        os.makedirs(group_anno_vis_dir, exist_ok=True)
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
                if annitem['type'] == 'circle':
                    ann_type[0] += 1
                if annitem['type'] == 'rect':
                    ann_type[1] += 1
                if annitem['type'] == 'polygon':
                    ann_type[2] += 1
                ann_clsnames.append(sub_class)
                if sub_class not in POSITIVE_CLASS:
                    beside_clsname_cnt += 1

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
                    'filename':f'{patientId}_{idx}.png',
                    'square_x1y1': (start_x, start_y),
                    'bboxes': [],
                    'clsnames': [],
                    'prefix': 'Neg',
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
                        patchinfo['prefix'] = 'Pos'

                if patchinfo['diagnose'] == 1:
                    roiInfo['patchlist'].append(patchinfo)
                elif patchinfo['diagnose'] == 0 and random.random() < 0.5:
                    roiInfo['patchlist'].append(patchinfo)

                # if downsample_ratio !=1:
                #     img = img.resize((imgw,imgh))
                # cropped = img.crop((sx1,sy1,sx2,sy2))
                # save_path = f'{group_anno_vis_dir}/random_{patchinfo["filename"]}'
                # inside_items = [dict(sub_class=clsname,region=region) for region,clsname in zip(patchinfo['bboxes'],patchinfo['clsnames'])]
                # draw_OD(cropped, save_path, [0,0,PATCH_EDGE,PATCH_EDGE],inside_items, POSITIVE_CLASS)

            total_roiInfo.append(roiInfo)
    # print(ann_type)
    # print(list(set(ann_clsnames)))
    # print(beside_clsname_cnt)
    return total_roiInfo

def foramt_json():
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

    new_total_roiInfo = []
    for roiInfo in tqdm(total_roiInfo, ncols=80, desc='Cuting img'):
        new_patch_list = []
        for patchinfo in roiInfo['patchlist']:
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
                print('error!!')
            else:
                new_patch_list.append(patchinfo)

        roiInfo['patchlist'] = new_patch_list
        new_total_roiInfo.append(roiInfo)

    with open('data_resource/0328/annofiles/zheyi_roi_4fusion_filtered.json','w') as f:
        json.dump(new_total_roiInfo, f)


if __name__ == '__main__':
    
    save_ann_json = 'data_resource/0328/annofiles/zheyi_roi_4fusion.json'
    
    # total_roiInfo = gene_ann_json()
    # analyze_patchlist(total_roiInfo)
    # with open(save_ann_json,'w') as f:
    #     json.dump(total_roiInfo, f)
    
    with open(save_ann_json,'r') as f:
        total_roiInfo = json.load(f)
    # cut_patchlist(total_roiInfo)
    foramt_json()
                

'''
PATCH_EDGE = 500, STRIDE = 450
Neg,Pos
- [8013, 4636]
[15359, 5708]
+ randomcut and select 50% neg patch
[Neg, Pos]
train: [8934, 6826]
val: [2298, 1807]
total: [11232, 8633]
+-------+-----+--------+------+-------+------+
|  Mode | AGC | ASC-US | LSIL | ASC-H | HSIL |
+-------+-----+--------+------+-------+------+
| Train |  68 |  4976  | 2534 |  2847 | 5056 |
|  Val  |  33 |  1609  | 618  |  948  | 1394 |
| Total | 101 |  6585  | 3152 |  3795 | 6450 |
+-------+-----+--------+------+-------+------+

PATCH_EDGE = 700, STRIDE = 650
Neg,Pos
[6686, 8831]
'''