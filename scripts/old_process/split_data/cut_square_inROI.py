import json
from tqdm import tqdm
import pandas as pd
import random
import numpy as np
from cerwsi.utils import KFBSlide
from multiprocessing import Pool
import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
import torch
from mmpretrain.structures import DataSample
from cerwsi.nets import ValidClsNet

POSITIVE_CLASS = ['ASC-US', 'LSIL', 'ASC-H', 'HSIL', 'SCC', 'AGC-NOS', 'AGC', 'AGC-N', 'AGC-FN']
colors = plt.cm.tab10(np.linspace(0, 1, len(POSITIVE_CLASS)))[:, :3] * 255
category_colors = {cat: tuple(map(int, color)) for cat, color in zip(POSITIVE_CLASS, colors)}

def draw_OD(read_image, save_path, square_coords, inside_items):
    draw = ImageDraw.Draw(read_image)
    sq_x1,sq_y1,sq_w,sq_h = square_coords

    for box_item in inside_items:
        category = box_item.get('sub_class')
        region = box_item.get('region')
        x,y = region['x'],region['y']
        w,h = region['width'],region['height']
        x1, y1, x2, y2 = x,y,x+w,y+h
        x_min = max(sq_x1, x1) - sq_x1
        y_min = max(sq_y1, y1) - sq_y1
        x_max = min(sq_x1+sq_w, x2) - sq_x1
        y_max = min(sq_y1+sq_h, y2) - sq_y1
        
        color = category_colors.get(category, (255, 255, 255))
        draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=3)
        draw.text((x_min + 2, y_min - 15), category, fill=color)
    
    # 使用 matplotlib 添加 legend
    fig, ax = plt.subplots(figsize=(sq_w//100+1, sq_h//100+1), dpi=100)
    ax.imshow(np.array(read_image))
    ax.axis('off')  # 不显示坐标轴
    # 创建 legend
    patches = [
        mpatches.Patch(color=np.array(color) / 255.0, label=category)  # Matplotlib 支持归一化颜色
        for category, color in category_colors.items()
    ]
    ax.legend(handles=patches, loc='upper right', bbox_to_anchor=(1.35, 1), frameon=False)
    fig.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

def trainval_split():
    train_ratio = 0.8
    pos_csv = pd.read_csv('data_resource/ROI/annofile/1223_pos.csv')
    neg_csv = pd.read_csv('data_resource/ROI/annofile/1223_neg.csv')
    df_data = pd.concat([pos_csv, neg_csv], ignore_index=True)

    pos_patientId = list(pos_csv['patientId'])
    neg_patientId = list(neg_csv['patientId'])
    random.shuffle(pos_patientId)
    random.shuffle(neg_patientId)
    train_pos_num = int(len(pos_patientId)*train_ratio)
    train_neg_num = int(len(neg_patientId)*train_ratio)
    train_patientId = [*pos_patientId[:train_pos_num], *neg_patientId[:train_neg_num]]
    val_patientId = [*pos_patientId[train_pos_num:], *neg_patientId[train_neg_num:]]

    # kfb_path,kfb_clsid,kfb_clsname,patientId,kfb_source
    for pids,mode in zip([train_patientId, val_patientId], ['train','val']):
        mode_data = []
        pos,neg = 0,0
        for pid in pids:
            patient_row = df_data.loc[df_data['patientId'] == pid].iloc[0]
            if patient_row.kfb_clsname == 'NILM':
                neg += 1
                mode_data.append([patient_row.kfb_path, 0, patient_row.kfb_clsname, pid, patient_row.kfb_source])
            else:
                pos += 1
                kfb_source = 'JFSW_1' if 'JFSW_1' in pid else 'JFSW_2'
                mode_data.append([patient_row.kfb_path, 1, patient_row.kfb_clsname, pid, kfb_source])
        mode_df = pd.DataFrame(mode_data, columns = ['kfb_path','kfb_clsid','kfb_clsname','patientId','kfb_source'])
        mode_df.to_csv(f'data_resource/ROI/annofile/1231_{mode}.csv', index=False)

        print(f'{mode} mode: train slide total {pos+neg}, positive is {pos} and negative is {neg}.')

def gene_annjson():
    dataroot = '/medical-data/data'
    device = torch.device('cuda:0')
    valid_model_ckpt = 'checkpoints/vlaid_cls_best.pth'
    valid_model = ValidClsNet()
    valid_model.to(device)
    valid_model.eval()
    valid_model.load_state_dict(torch.load(valid_model_ckpt))
    CERTAIN_THR = 0.7

    classes = ['negative', 'ASC-US', 'LSIL', 'ASC-H', 'HSIL', 'AGC']
    categories = [{'category_name':clsname, 'id':clsid} for clsid,clsname in enumerate(classes)]
    # 类别映射关系
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

    json_path = 'data_resource/ROI/annofile/square_in_pos_v2.json'
    with open(json_path, 'r') as f:
        squares = json.load(f)
    
    train_csv = pd.read_csv('data_resource/ROI/annofile/1231_train.csv')
    train_patientId = list(train_csv['patientId'])

    pos_csv = pd.read_csv('data_resource/ROI/annofile/1231_pos.csv')

    metainfo = {'categories':categories}
    train_datalist,val_datalist = [], []
    cnts = [0,0,0,0]    # train: neg, pos   val: neg,pos
    for idx,squ_item in enumerate(tqdm(squares['square_items'], ncols=80)):
        # if idx < 201:
        #     continue
        img_path = f'images/{squ_item["filename"]}'
        patientId = squ_item["patientId"]
        pos_anns = squ_item["pos_anns"]
        
        gt_label = None
        if len(pos_anns) > 0:
            gt_label = []
            for annitem in pos_anns:
                ann_clsname = RECORD_CLASS[annitem['sub_class']]
                ann_clsid = classes.index(ann_clsname)
                gt_label.append(ann_clsid)
            gt_label = list(set(gt_label))
        else:
            patient_row = pos_csv.loc[pos_csv['patientId'] == patientId].iloc[0]
            slide = KFBSlide(f"{dataroot}/{patient_row['kfb_path']}")
            x1,y1 = squ_item['square_x1y1']
            location, level, size = (x1,y1), 0, (500,500)
            read_result = Image.fromarray(slide.read_region(location, level, size))
            img_input = cv2.cvtColor(np.array(read_result), cv2.COLOR_RGB2BGR)
            img_input = torch.as_tensor(cv2.resize(img_input, (224,224)))
            data_batch = dict(inputs=[], data_samples=[])
            data_batch['inputs'].append(img_input.permute(2,0,1))    # (bs, 3, h, w)
            data_batch['data_samples'].append(DataSample())
            data_batch['inputs'] = torch.stack(data_batch['inputs'], dim=0)
            with torch.no_grad():
                outputs = valid_model.val_step(data_batch)
            if max(outputs[0].pred_score) > CERTAIN_THR and outputs[0].pred_label == 1:
                gt_label = [0]
        
        if gt_label is not None:
            ann_item = {"img_path": img_path,"gt_label": gt_label,'squ_item':squ_item}
            if patientId in train_patientId:
                idx = 1 if len(pos_anns) > 0 else 0
                cnts[idx] += 1
                train_datalist.append(ann_item)
            else:
                idx = 3 if len(pos_anns) > 0 else 2
                cnts[idx] += 1
                val_datalist.append(ann_item)
    
    train_ann_dict = dict(metainfo=metainfo, data_list=train_datalist)
    val_ann_dict = dict(metainfo=metainfo, data_list=val_datalist)
    with open('data_resource/ROI/annofile/1231_train_ann.json', 'w') as f:
        json.dump(train_ann_dict, f)
    with open('data_resource/ROI/annofile/1231_val_ann.json', 'w') as f:
        json.dump(val_ann_dict, f)
    print(f'train neg/pos and val neg/pos: {cnts}')

def process_patches(proc_id, pid2square_dict):
    data_root_dir = '/medical-data/data'
    img_savedir = 'data_resource/ROI/square_data/images_v2'
    imgOD_savedir = 'data_resource/ROI/imagesOD_v2'
    os.makedirs(img_savedir, exist_ok=True)
    os.makedirs(imgOD_savedir, exist_ok=True)
    WINDOW_SIZE = 500
    df_data = pd.read_csv('data_resource/ROI/annofile/1231_pos.csv')

    processing = 0
    for pid,squares in pid2square_dict.items():
        patient_row = df_data.loc[df_data['patientId'] == pid].iloc[0]
        slide = KFBSlide(f'{data_root_dir}/{patient_row.kfb_path}')
        for squ_item in squares:
            save_filename = squ_item["filename"]
            square_x1y1 = squ_item['square_x1y1']
            location, level, size = square_x1y1, 0, (WINDOW_SIZE,WINDOW_SIZE)
            read_result = Image.fromarray(slide.read_region(location, level, size))
            read_result.save(f'{img_savedir}/{save_filename}')
            
            square_clsid = 0 if squ_item['pos_nums'] == 0 else 1
            if square_clsid == 1 and random.random() < 0.001:
                save_path = f'{imgOD_savedir}/{save_filename}'
                draw_OD(read_result, save_path, [*square_x1y1, WINDOW_SIZE,WINDOW_SIZE], squ_item['pos_anns'])
        processing += 1
        print(f'\rCore: {proc_id}, 当前已处理: {processing}/{len(pid2square_dict.keys())}', end='')

def split_dict_items(data: dict, k: int):

    items = list(data.items())
    n = len(items)
    group_size = n // k
    remainder = n % k
    result = []
    start = 0
    for i in range(k):
        # 每组大小：基础长度 + 是否包含一个余数
        size = group_size + (1 if i < remainder else 0)
        result.append(items[start:start + size])
        start += size
    return [dict(group) for group in result]

def cut_square():
    json_path = 'data_resource/ROI/annofile/square_in_pos.json'
    with open(json_path, 'r') as f:
        squares = json.load(f)

    pid2squares = dict()
    for squ_item in tqdm(squares['square_items'], ncols=80):
        pid = squ_item['patientId']
        pid2squares.setdefault(pid, []).append(squ_item)

    cpu_num = 8
    set_split = split_dict_items(pid2squares, cpu_num)
    print(f"Number of cores: {cpu_num}, set number of per core: {len(set_split[0])}")
    workers = Pool(processes=cpu_num)
    processes = []
    for proc_id, set_group in enumerate(set_split):
        p = workers.apply_async(process_patches, (proc_id, set_group,))
        processes.append(p)
    for p in processes:
        p.get()
    workers.close()
    workers.join()
 
def cut_square_v2():
    train_json_path = 'data_resource/ROI/annofile/1231_train_ann.json'
    val_json_path = 'data_resource/ROI/annofile/1231_val_ann.json'
    with open(train_json_path, 'r') as f:
        train_json = json.load(f)
    with open(val_json_path, 'r') as f:
        val_json = json.load(f)

    pid2squares = dict()
    total_squares = [*train_json['data_list'],*val_json['data_list']]
    for dataitem in tqdm(total_squares, ncols=80):
        squ_item = dataitem['squ_item']
        pid = squ_item['patientId']
        pid2squares.setdefault(pid, []).append(squ_item)

    cpu_num = 8
    set_split = split_dict_items(pid2squares, cpu_num)
    print(f"Number of cores: {cpu_num}, set number of per core: {len(set_split[0])}")
    workers = Pool(processes=cpu_num)
    processes = []
    for proc_id, set_group in enumerate(set_split):
        p = workers.apply_async(process_patches, (proc_id, set_group,))
        processes.append(p)
    for p in processes:
        p.get()
    workers.close()
    workers.join()

def statistic_patch():
    train_json_path = 'data_resource/ROI/annofile/1231_train_ann.json'
    val_json_path = 'data_resource/ROI/annofile/1231_val_ann.json'
    with open(train_json_path, 'r') as f:
        train_json = json.load(f)
    with open(val_json_path, 'r') as f:
        val_json = json.load(f)
    classes = ['negative', 'ASC-US', 'LSIL', 'ASC-H', 'HSIL', 'AGC']

    for datalist,mode in zip([train_json['data_list'], val_json['data_list']],['train','val']):
        pos,neg = 0,0
        pos_cls = [0] * len(classes)
        for dataitem in tqdm(datalist, ncols=80):
            squ_item = dataitem['squ_item']
            if squ_item['pos_nums']>0:
                pos += 1
                gt_label = dataitem['gt_label']
                for clsid in gt_label:
                    pos_cls[clsid] += 1
            else:
                neg += 1
        print(f'{mode}: neg {neg} pos {pos} sub class {pos_cls}')


if __name__ == '__main__':
    # trainval_split()
    # gene_annjson()
    # cut_square_v2()
    statistic_patch()


'''
train mode: train slide total 1581, positive is 816 and negative is 765.
val mode: train slide total 397, positive is 205 and negative is 192.
'''