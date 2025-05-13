import multiprocessing
from tqdm import tqdm
import argparse
import numpy as np
import pandas as pd
import warnings
import os
import random
import json
from multiprocessing import Pool
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import torch
from mmpretrain.structures import DataSample
from cerwsi.utils import (KFBSlide,remap_points,read_json_anno,decode_xml,is_bbox_inside,calc_relative_coord,draw_OD)

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.conv")

CERTAIN_THR = 0.7
LEVEL = 0
PATCH_EDGE = 224
STRIDE = 200
NEGATIVE_CLASS = ['NILM', 'GEC']
POSITIVE_CLASS = ['ASC-US', 'LSIL', 'ASC-H', 'HSIL', 'SCC', 'AGC-NOS', 'AGC', 'AGC-N', 'AGC-FN']


def find_matching_bboxes(target_bbox, grid_size = PATCH_EDGE, stride = STRIDE, min_overlap=50):
    matching_bboxes = []
    x1_min, y1_min, x1_max, y1_max = target_bbox

    # 计算col 和 row 范围
    col_min = int(max(0, x1_min // stride))
    col_max = int(x1_max // stride)
    row_min = int(max(0, y1_min // stride))
    row_max = int(y1_max // stride)
    
    for row in range(row_min, row_max + 1):
        for col in range(col_min, col_max + 1):
            x2_min = col * stride
            y2_min = row * stride
            x2_max = x2_min + grid_size
            y2_max = y2_min + grid_size
            bbox = [x2_min, y2_min, x2_max, y2_max]
            # 判断完全包含 (target_bbox完全在该patch内部)
            if is_bbox_inside(target_bbox, bbox):
                relative_bbox = [x1_min - x2_min, y1_min - y2_min, x1_max - x2_min, y1_max - y2_min]
                matching_bboxes.append(([row,col], relative_bbox))
                continue

            # 计算交集区域
            inter_x_min = max(x1_min, x2_min)
            inter_y_min = max(y1_min, y2_min)
            inter_x_max = min(x1_max, x2_max)
            inter_y_max = min(y1_max, y2_max)

            if inter_x_min < inter_x_max and inter_y_min < inter_y_max:
                inter_w = inter_x_max - inter_x_min
                inter_h = inter_y_max - inter_y_min
                if inter_w > min_overlap and inter_h > min_overlap:
                    relative_bbox = [inter_x_min - x2_min, inter_y_min - y2_min, inter_x_max - x2_min, inter_y_max - y2_min]
                    matching_bboxes.append(([row,col], relative_bbox))

    return matching_bboxes

def process_pos_slide(rowInfo):
    pos_patches_result = {} # key is patch id, value is patch anno info
    slide = KFBSlide(rowInfo.kfb_path)
    swidth, sheight = slide.level_dimensions[LEVEL]
    downsample_ratio = slide.level_downsamples[LEVEL]
    total_cols = int(swidth // STRIDE) + 1
    json_path = f'{args.data_root_dir}/{rowInfo.anno_path}'
    annos = read_json_anno(json_path)
    
    for ann_ in annos:
        ann = remap_points(ann_)
        if ann is None:
            continue
        sub_class = ann.get('sub_class')
        region = ann.get('region')
        x,y = region['x'],region['y']
        w,h = region['width'],region['height']

        if w <=20 or h<=20 or sub_class not in [*NEGATIVE_CLASS, *POSITIVE_CLASS]:
            continue

        scaled_bbox = np.array([x,y,x+w,y+h]) // downsample_ratio
        target_bbox = scaled_bbox.tolist()
        match_patches = find_matching_bboxes(target_bbox, min_overlap=25)
        for stp,coord in match_patches:
            row,col= stp
            patchid = row * total_cols + col
            sx1,sy1 = col * STRIDE, row * STRIDE
            sx2,sy2 = sx1 + PATCH_EDGE, sy1 + PATCH_EDGE
            if sx2 > swidth:
                exceed = sx2 - swidth
                sx1 = sx1 - exceed
                sx2 = swidth
                coord = calc_relative_coord([sx1,sy1,sx2,sy2],target_bbox)
            if sy2 > sheight:
                exceed = sy2 - sheight
                sy1 = sy1 - exceed
                sy2 = sheight
                coord = calc_relative_coord([sx1,sy1,sx2,sy2],target_bbox)

            if patchid not in pos_patches_result.keys():
                pos_patches_result[patchid] = {
                    'filename':f'{rowInfo.patientId}_{patchid}.png',
                    'square_x1y1': (sx1,sy1),
                    'bboxes': [coord],
                    'clsnames': [sub_class],
                    'prefix': 'partial_pos',
                    'diagnose': 1
                }
            else:
                pos_patches_result[patchid]['bboxes'].append(coord)
                pos_patches_result[patchid]['clsnames'].append(sub_class)
    
    pos_patch_list = filter_patch_list(pos_patches_result.values())
    
    # 确保每个patch中至少有一个阳性标注框
    keep_patches = []
    for patchInfo in pos_patch_list:
        unique_clsnames = list(set(patchInfo['clsnames']))
        for clsname in unique_clsnames:
            if clsname in POSITIVE_CLASS:
                keep_patches.append(patchInfo)
                break
            
    return keep_patches

def process_pos_slide_wxl(rowInfo):
    patches_result = {} # key is patch id, value is patch anno info
    slide = KFBSlide(rowInfo.kfb_path)
    swidth, sheight = slide.level_dimensions[LEVEL]
    downsample_ratio = slide.level_downsamples[LEVEL]
    total_cols = int(swidth // STRIDE) + 1
    
    anno_path = f'{args.data_root_dir}/{rowInfo.anno_path}'
    all_rects = decode_xml(anno_path)
    for x1,y1,x2,y2 in all_rects:
        w,h = x2-x1, y2-y1
        sub_class = rowInfo.kfb_clsname

        if w <=20 or h<=20:
            continue
        scaled_bbox = np.array([x1,y1,x2,y2]) // downsample_ratio
        target_bbox = scaled_bbox.tolist()
        match_patches = find_matching_bboxes(target_bbox, min_overlap=25)
        for stp,coord in match_patches:
            row,col= stp
            patchid = row * total_cols + col
            sx1,sy1 = col * STRIDE, row * STRIDE
            sx2,sy2 = sx1 + PATCH_EDGE, sy1 + PATCH_EDGE
            if sx2 > swidth:
                exceed = sx2 - swidth
                sx1 = sx1 - exceed
                sx2 = swidth
                coord = calc_relative_coord([sx1,sy1,sx2,sy2],target_bbox)
            if sy2 > sheight:
                exceed = sy2 - sheight
                sy1 = sy1 - exceed
                sy2 = sheight
                coord = calc_relative_coord([sx1,sy1,sx2,sy2],target_bbox)

            if patchid not in patches_result.keys():
                patches_result[patchid] = {
                    'filename':f'{rowInfo.patientId}_{patchid}.png',
                    'square_x1y1': (sx1,sy1),
                    'bboxes': [coord],
                    'clsnames': [sub_class],
                    'prefix': 'partial_pos',
                    'diagnose': 1
                }
            else:
                patches_result[patchid]['bboxes'].append(coord)
                patches_result[patchid]['clsnames'].append(sub_class)
    pos_patch_list = filter_patch_list(list(patches_result.values()))
    return pos_patch_list

def filter_roi_slide():
    invalid_pids = [
        'JFSW_1_57', 'JSFW_1_67', 'JSFW_2_125', 'JSFW_2_1323', 'JSFW_2_1327', 'JSFW_2_1583', 'JSFW_2_1521',
        'JFSW_2_837',
        'JFSW_2_1308', 'JFSW_2_255', 'JFSW_2_360',
    ]
    ann_json = 'data_resource/0328/annofiles/宫颈液基细胞—RoI.json'
    with open(ann_json,'r') as f:
        annotaion = json.load(f)

    Group1 = annotaion['Group1']
    Group3 = annotaion['Group3']
    Group4 = annotaion['Group4']

    filter_patientIds = []
    for itemlist,tag in zip([Group1,Group3,Group4],['group1','group3','group4']):
        for imgitem in tqdm(itemlist, ncols=80):
            filename = imgitem['imageName']
            patientId = filename.replace('_RoI.png','')
            if patientId in invalid_pids:
                continue
            filter_patientIds.append(patientId)
    return filter_patientIds

def gene_patch_json():
    train_data_df = pd.read_csv(args.train_csv_file)
    val_data_df = pd.read_csv(args.val_csv_file)

    for data_df,mode in zip([train_data_df,val_data_df], ['train','val']):
        all_patch_list = []
        total_pos_nums = 0
        for row in tqdm(data_df.itertuples(index=True), total=len(data_df), ncols=80):
            # if row.Index > 5:
            #     break
            if row.kfb_clsname == 'NILM' or row.kfb_source == 'ZY_ONLINE_1':
                continue
            if row.kfb_source == 'WXL_1':
                pos_patch_list = process_pos_slide_wxl(row)
            else:
                pos_patch_list = process_pos_slide(row)

            total_pos_nums += len(pos_patch_list)
            all_patch_list.append({
                'patientId': row.patientId,
                'kfb_path': row.kfb_path,
                'patch_list': pos_patch_list
            })
        print(f'{mode}: {total_pos_nums} pos patches.')

        with open(f'{ann_save_dir}/partial_{mode}_pos_{PATCH_EDGE}.json', 'w') as f:
            json.dump(all_patch_list, f)
    

def cut_save(kfb_list):
    from cerwsi.nets import ValidClsNet

    device = torch.device('cuda:0')
    valid_model_ckpt = 'checkpoints/valid_cls_best.pth'
    valid_model = ValidClsNet()
    valid_model.to(device)
    valid_model.eval()
    valid_model.load_state_dict(torch.load(valid_model_ckpt))

    img_save_dir = f'data_resource/0429/{PATCH_EDGE}/images/partial_pos'
    os.makedirs(img_save_dir, exist_ok=True)

    for kfbinfo in tqdm(kfb_list, ncols=80):
        slide = KFBSlide(kfbinfo["kfb_path"])
        patch_list = kfbinfo['patch_list']
        for patchinfo in patch_list:
            x1,y1 = patchinfo['square_x1y1']
            location, level, size = (x1,y1), LEVEL, (PATCH_EDGE,PATCH_EDGE)
            read_result = Image.fromarray(slide.read_region(location, level, size))
            data_batch = dict(inputs=[], data_samples=[])
            img_input = cv2.cvtColor(np.array(read_result), cv2.COLOR_RGB2BGR)
            img_input = torch.as_tensor(cv2.resize(img_input, (224,224)))
            data_batch['inputs'].append(img_input.permute(2,0,1))    # (bs, 3, h, w)
            data_batch['data_samples'].append(DataSample())
            data_batch['inputs'] = torch.stack(data_batch['inputs'], dim=0)
            with torch.no_grad():
                outputs = valid_model.val_step(data_batch)
            if max(outputs[0].pred_score) > CERTAIN_THR and outputs[0].pred_label == 1:
                read_result.save(f'{img_save_dir}/{patchinfo["filename"]}')

def cut_patch():

    for mode in ['train', 'val']:
        with open(f'{ann_save_dir}/partial_{mode}_pos_{PATCH_EDGE}.json', 'r') as f:
            kfb_list = json.load(f)
        
        cpu_num = 8
        set_split = np.array_split(kfb_list, cpu_num)
        print(f"Number of cores: {cpu_num}, set number of per core: {len(set_split[0])}")
        workers = Pool(processes=cpu_num)
        processes = []
        for proc_id, set_group in enumerate(set_split):
            p = workers.apply_async(cut_save,(set_group,))
            processes.append(p)

        for p in processes:
            p.get()

def vis_sample(slide,patchinfo):
    sample_save_dir = 'statistic_results/0403/filter_pos_sample'
    os.makedirs(sample_save_dir, exist_ok=True)
    x1,y1 = patchinfo['square_x1y1']
    innerbbox,bbox_clsname = patchinfo['bboxes'],patchinfo['clsnames']

    inside_items = []
    for coords,clsname in zip(innerbbox,bbox_clsname):
        inside_items.append({'sub_class': clsname,'region': coords})
    
    location, level, size = (x1,y1), LEVEL, (PATCH_EDGE,PATCH_EDGE)
    read_result = Image.fromarray(slide.read_region(location, level, size))
    filename = patchinfo["filename"]
    square_coords = [0,0,PATCH_EDGE,PATCH_EDGE]
    draw_OD(read_result, f'{sample_save_dir}/{filename}', square_coords, inside_items, POSITIVE_CLASS)

def filter_patch_list(kfb_patch_list):
    '''
    1. 去掉阳性框内部的小框
    2. 若阴性框内部有阳性框，丢弃阴性框保留阳性框
    '''

    def deduplicate_bboxes(bboxes, clsnames, tolerance=5):
        """
        去重 bbox 列表，允许坐标有小误差（比如5px内）
        
        Args:
            bboxes: (n,4) numpy array，格式是 [x1, y1, x2, y2]
            tolerance: 允许的坐标误差
        Returns:
            filtered_bboxes: 去重后的 bbox numpy array
        """
        bboxes = np.array(bboxes)
        kept_bboxes,kept_clsname = [],[]

        for bbox,cname in zip(bboxes, clsnames):
            is_duplicate = False
            for kept_bbox in kept_bboxes:
                if np.all(np.abs(bbox - kept_bbox) <= tolerance):
                    is_duplicate = True
                    break
            if not is_duplicate:
                kept_bboxes.append(bbox)
                kept_clsname.append(cname)
        
        return kept_bboxes,kept_clsname

    def pos_box_enclosured(bbox, all_bboxes):
        for parent_bbox in all_bboxes:
            if bbox[0] == parent_bbox[0] and bbox[1] == parent_bbox[1] and bbox[2] == parent_bbox[2] and bbox[3] == parent_bbox[3]:
                continue
            if is_bbox_inside(bbox, parent_bbox, 5):
                return True
        return False
    
    def has_posbox_inside(bbox, all_bboxes):
        for child_bbox in all_bboxes:
            if is_bbox_inside(child_bbox, bbox, 5):
                return True
        return False

    new_patch_list = []
    for patchinfo in kfb_patch_list:
        if len(patchinfo['bboxes']) != 1:
            new_bboxes,nes_clsnames = [],[]
            filtered_bboxes,filtered_clsnames = deduplicate_bboxes(patchinfo['bboxes'], patchinfo['clsnames'], tolerance=5)
            posboxes = [box for box,clsn in zip(filtered_bboxes,filtered_clsnames) if clsn in POSITIVE_CLASS]
            for bbox,clsname in zip(filtered_bboxes,filtered_clsnames):
                if clsname in POSITIVE_CLASS and pos_box_enclosured(bbox, posboxes):
                    continue
                if clsname in NEGATIVE_CLASS and has_posbox_inside(bbox, posboxes):
                    continue
                new_bboxes.append(bbox.tolist())
                nes_clsnames.append(clsname)
            patchinfo['bboxes'] = new_bboxes
            patchinfo['clsnames'] = nes_clsnames
        
        if len(patchinfo['bboxes']) != 0:
            new_patch_list.append(patchinfo)

    return new_patch_list


parser = argparse.ArgumentParser()
parser.add_argument('train_csv_file', type=str)
parser.add_argument('val_csv_file', type=str)
parser.add_argument('--data_root_dir', type=str, default='/medical-data/data')

args = parser.parse_args()

if __name__ == '__main__':
    
    ann_save_dir = 'data_resource/0429/annofiles'
    img_save_dir = f'data_resource/0429/{PATCH_EDGE}/images/partial_pos'
    os.makedirs(img_save_dir, exist_ok=True)
    # gene_patch_json()
    multiprocessing.set_start_method('spawn', force=True)
    cut_patch()


'''
python scripts/0429/partial_pos_slide_cut.py \
    data_resource/0429/annofiles/partial_train.csv \
    data_resource/0429/annofiles/partial_val.csv

512
train: 46947 pos patches.  →  46460
val: 15128 pos patches.  →  14957
total: 62075 pos patches.


224
train: 78887 pos patches.  →  76682
val: 27315 pos patches.  →  26513
total: 106202 pos patches.
'''