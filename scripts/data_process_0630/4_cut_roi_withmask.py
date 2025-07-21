import os
from PIL import Image
import json
import pandas as pd
from scipy import sparse
from cerwsi.utils import KFBSlide,random_cut_square,calc_relative_coord,is_bbox_inside,generate_cut_regions
from tqdm import tqdm
import numpy as np
import warnings
import cv2
import torch
from scipy.ndimage import find_objects
import matplotlib.pyplot as plt
from collections import defaultdict
from mmpretrain.structures import DataSample
import multiprocessing
from multiprocessing import Pool
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.conv")

CERTAIN_THR = 0.7
LEVEL = 0
WINDOW_SIZE = 1600
STRIDE = WINDOW_SIZE - 50

test_patientId = [
    'ZY_ONLINE_1_45', 'ZY_ONLINE_1_21', 'ZY_ONLINE_1_1481',   # 0409 Slide，0422 Slide, 0607 Slide
    # 'JFSW_2_1486', 'JFSW_2_152',     # 空 RoI，有标注 RoI
    # 'JFSW_1_2', 'JFSW_2_2111', 'WXL_1_26'     # partial_pos slide 'JFSW_2_133', 
]

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image) 

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  

def vis_patch_sample(image, roi_masks, bboxes, filename):
    plt.figure(figsize=(6, 6))
    plt.imshow(image)

    for box in bboxes:
        show_box(box, plt.gca())
    
    annids = np.unique(roi_masks)
    for annid_idx in annids[1:]:   # 第一个是 0
        mask = roi_masks == annid_idx
        show_mask(mask, plt.gca(), random_color=True)
        
    plt.axis('off')
    plt.tight_layout()
    os.makedirs('statistic_results/0630/patch_from_RoI', exist_ok=True, mode=0o777)
    plt.savefig(f'statistic_results/0630/patch_from_RoI/{filename}')
    plt.close()

def gene_patch_jsonlist(proc_id, all_json_datas, npz_mask_save_dir, patch_npz_save_dir):
    patchitems = []
    for idx, item in enumerate(all_json_datas):
        # if item['patientId'] not in test_patientId:
        #     continue
        for RoIItem in item['annotations']:
            rx1,ry1,rx2,ry2 = (np.array(RoIItem['region']).astype(np.int32)).tolist()
            rw,rh = rx2-rx1, ry2-ry1
            purename = item['patientId'] + f'_{str(RoIItem["annid"])}'
            
            if len(RoIItem['children']) > 0:
                loader = np.load(f"{npz_mask_save_dir}/{purename}.npz")
                sparse_mask = sparse.coo_matrix((loader['data'], (loader['row'], loader['col'])), shape=loader['shape'])
                roi_mask = sparse_mask.toarray().astype(np.int16)
            else:
                roi_mask = np.zeros((rh,rw), dtype=np.int16)

            RoI_patch_idx = 0
            if abs(rw-WINDOW_SIZE) < 100 or abs(rh-WINDOW_SIZE) < 100:
                random_x1,random_y1 = random_cut_square([0,0,rw,rh],WINDOW_SIZE)  # 在 RoI 中的相对坐标
                cut_points = [(random_x1,random_y1)]
            else:
                cut_points = generate_cut_regions((0,0), rw-1, rh-1, WINDOW_SIZE, STRIDE)

            # random_cut_num = 20 if rw > 5000 and rh > 5000 else 5
            # for i in range(random_cut_num):
            #     random_x1,random_y1 = random_cut_square([0,0,rw,rh],WINDOW_SIZE)  # 在 RoI 中的相对坐标
            #     random_x2,random_y2 = random_x1+WINDOW_SIZE,random_y1+WINDOW_SIZE
            #     if random_x2 < rw and random_y2 < rh:
            #         cut_points.append((random_x1,random_y1))

            # 按照bbox area 从大到小排序
            RoIItem['children'] = sorted(
                RoIItem['children'],
                key=lambda annitem: (annitem['region'][2] - annitem['region'][0]) * (annitem['region'][3] - annitem['region'][1]),
                reverse=True  # 从大到小排序
            )
            for iidx,rect_coords in enumerate(cut_points):
                x1,y1 = rect_coords
                x2,y2 = x1+WINDOW_SIZE,y1+WINDOW_SIZE
                patch_coords = [x1,y1,x2,y2]    # 在 RoI 中的相对坐标
                bboxes,clsnames,patch_mask = calc_patch_anns(patch_coords, RoIItem, roi_mask)
                if patch_mask.shape[0] != WINDOW_SIZE or patch_mask.shape[1] != WINDOW_SIZE:
                    print(f'Shape not right: {patch_mask.shape}')
                if len(bboxes) == 0 and RoIItem['sub_class'] == 'forge_RoI':
                    continue

                pItem = {
                    'patientId': item['patientId'],
                    'media_type': item['media_type'],
                    'source_path': item['source_path'],
                    'square_coords': [x1+rx1,y1+ry1,x2+rx1,y2+ry1],    # 在媒体资源中的相对坐标，用以切图片
                    'filename': f'{purename}_{RoI_patch_idx}.png',
                    'bboxes': bboxes,    # 在patch中的相对坐标
                    'clsnames': clsnames,
                }
                px1,py1,px2,py2 = pItem['square_coords']
                if (px2-px1) != WINDOW_SIZE or (py2-py1) != WINDOW_SIZE:
                    print(f'Shape not right: {pItem["square_coords"]}')
                
                if len(bboxes) == 0:
                    pItem['prefix'] = 'neg'
                    pItem['diagnose'] = 0
                    pItem['maskfile'] = ''
                elif len(bboxes) > 0:
                    pItem['prefix'] = 'total_pos' if RoIItem['sub_class'] == 'RoI' else 'partial_pos'
                    pItem['diagnose'] = 1
                    pItem['maskfile'] = f'{purename}_{RoI_patch_idx}.npz'
                    np.savez_compressed(f'{patch_npz_save_dir}/{pItem["maskfile"]}', patch_mask=patch_mask)
                
                    # slide = KFBSlide(pItem['source_path'])
                    # px1,py1,px2,py2 = pItem['square_coords']
                    # location, level, size = (px1,py1), 0, (px2-px1,py2-py1)
                    # patch_img = Image.fromarray(slide.read_region(location, level, size))
                    # vis_patch_sample(patch_img, patch_mask, bboxes, pItem['filename'])

                patchitems.append(pItem)
                RoI_patch_idx += 1

        print(f'Core {proc_id} processed : {idx+1}/{len(all_json_datas)}.')
    return patchitems

def calc_patch_anns(patch_coords, RoIItem, roi_mask):
    rpx1, rpy1, rpx2, rpy2 = patch_coords  # 相对 ROI 的坐标
    patch_mask = roi_mask[rpy1:rpy2, rpx1:rpx2]
    new_patch_mask = np.zeros_like(patch_mask, dtype=np.uint8)
    
    ann_bboxes, ann_clsnames = [], []
    annidx = np.unique(patch_mask)
    if len(annidx) <= 1:
        return ann_bboxes, ann_clsnames, new_patch_mask  # only background (0)

    # 使用 find_objects 获取所有非零实例的切片区域
    objects = find_objects(patch_mask)  # 返回一个 列表 slices，长度等于 input 中的最大 label 值（不包括 0）
    for aidx in annidx[1:]:  # 跳过背景 0
        obj_slice = objects[aidx-1]
        if obj_slice is None:
            continue
        yslice, xslice = obj_slice
  
        by1, by2 = yslice.start, yslice.stop
        bx1, bx2 = xslice.start, xslice.stop
        bwidth, bheight = bx2 - bx1, by2 - by1
        
        # 判断是否为贴边小目标
        is_small = min(bwidth, bheight) < 50
        is_near_edge = (
            bx1 <= 1 or by1 <= 1 or
            bx2 >= patch_mask.shape[1] - 1 or
            by2 >= patch_mask.shape[0] - 1
        )
        if is_small and is_near_edge:
            continue  # 丢弃该 annitem
        
        ann_bboxes.append([bx1, by1, bx2, by2])
        annitem = RoIItem['children'][aidx - 1]
        ann_clsnames.append(annitem['sub_class'])
        region = (patch_mask[yslice, xslice] == aidx)
        new_patch_mask[yslice, xslice][region] = len(ann_bboxes)  # 实例 ID 从 1 开始

    return ann_bboxes, ann_clsnames, new_patch_mask

def process_cut(proc_id, img_save_dir, patchlist:dict):
    from cerwsi.nets import ValidClsNet

    device = torch.device('cuda:0')
    valid_model_ckpt = 'checkpoints/valid_cls_best.pth'
    valid_model = ValidClsNet()
    valid_model.to(device)
    valid_model.eval()
    valid_model.load_state_dict(torch.load(valid_model_ckpt))
    total_nums = len(patchlist.keys())
    valid_patches = []
    for (keyname, patchlist),idx in zip(patchlist.items(), range(total_nums)):
        source_path = patchlist[0]['source_path']
        media_type = patchlist[0]['media_type']
        if media_type == 'roi':
            roi_img = Image.open(source_path)
        elif media_type == 'slide':
            slide = KFBSlide(source_path)
        for patchinfo in patchlist:
            px1,py1,px2,py2 = patchinfo['square_coords']
            w,h = px2-px1,py2-py1
            if w!=WINDOW_SIZE or h!=WINDOW_SIZE:
                print(f'ERROR: {patchinfo["filename"]} size (w,h): ({w},{h})')

            if media_type == 'roi':
                patch_img = roi_img.crop(patchinfo['square_coords'])
            elif media_type == 'slide':
                px1,py1,px2,py2 = patchinfo['square_coords']
                location, level, size = (px1,py1), 0, (px2-px1,py2-py1)
                patch_img = Image.fromarray(slide.read_region(location, level, size))

            if patchinfo['prefix'] == 'neg':    # 没有阳性框的patch需要判定是否为有效patch
                data_batch = dict(inputs=[], data_samples=[])
                img_input = cv2.cvtColor(np.array(patch_img), cv2.COLOR_RGB2BGR)
                img_input = torch.as_tensor(cv2.resize(img_input, (224,224)))
                data_batch['inputs'].append(img_input.permute(2,0,1))    # (bs, 3, h, w)
                data_batch['data_samples'].append(DataSample())
                data_batch['inputs'] = torch.stack(data_batch['inputs'], dim=0)
                with torch.no_grad():
                    outputs = valid_model.val_step(data_batch)
                if max(outputs[0].pred_score) > CERTAIN_THR and outputs[0].pred_label == 1:
                    patch_img.save(f'{img_save_dir}/neg/{patchinfo["filename"]}')
                    valid_patches.append(patchinfo)
            else:
                patch_img.save(f'{img_save_dir}/{patchinfo["prefix"]}/{patchinfo["filename"]}')
                valid_patches.append(patchinfo)
                # patch_mask = np.load(f'{patch_npz_save_dir}/{patchinfo["maskfile"]}')['patch_mask']
                # vis_patch_sample(patch_img, patch_mask, patchinfo['bboxes'], patchinfo['filename'])
        
        print(f'Core {proc_id} processed : {idx+1}/{total_nums}.')
    return valid_patches

def cut_patch_imgs(img_save_dir):
    with open(f'{json_save_dir}/{patches_jsonname}.json', 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    reload_patchlist = defaultdict(list)
    for patchinfo in tqdm(json_data, ncols=80):
        keyname = f"{patchinfo['patientId']}_{patchinfo['media_type']}"
        reload_patchlist[keyname].append(patchinfo)

    total_nums = len(reload_patchlist.keys())
    cpu_num = 8
    set_split = np.array_split(range(total_nums), cpu_num)
    print(f"Number of cores: {cpu_num}, total_nums: {total_nums}, set number of per core: {len(set_split[0])}")
    workers = Pool(processes=cpu_num)
    processes = []
    keys = list(reload_patchlist.keys())
    for proc_id, set_group in enumerate(set_split):
        process_group = {}
        for k in [keys[i] for i in set_group]:
            process_group[k] = reload_patchlist[k]
        p = workers.apply_async(process_cut, (proc_id, img_save_dir, process_group))
        processes.append(p)
    valid_patches_in_RoI = []
    for p in processes:
        valid_results = p.get()
        valid_patches_in_RoI.extend(valid_results)
    workers.close()
    workers.join()

    with open(f'{json_save_dir}/{patches_jsonname}_valid.json', 'w', encoding='utf-8') as f:
        json.dump(valid_patches_in_RoI, f, ensure_ascii=False)

def statistic_imgs():
    with open(f'{json_save_dir}/{patches_jsonname}_valid.json', 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    pn_cnt = [0,0,0]
    for patchinfo in tqdm(json_data, ncols=80):
        idx = -1
        if patchinfo['prefix'] == 'neg':
            idx = 0
        elif patchinfo['prefix'] == 'total_pos':
            idx = 1
        elif patchinfo['prefix'] == 'partial_pos':
            idx = 2
        pn_cnt[idx] += 1
    print(pn_cnt)

if __name__ == "__main__":
    timetag = '0630'
    npz_mask_save_dir = f'data_resource/{timetag}/roi_inst_mask'
    img_save_dir = f'data_resource/{timetag}/WINDOW_SIZE_{WINDOW_SIZE}/images'
    patch_npz_save_dir = f'data_resource/{timetag}/WINDOW_SIZE_{WINDOW_SIZE}/patch_inst_mask'
    os.makedirs(patch_npz_save_dir, exist_ok=True, mode=0o777)
    json_save_dir = f'data_resource/{timetag}/WINDOW_SIZE_{WINDOW_SIZE}/ann_jsons'
    os.makedirs(json_save_dir, exist_ok=True, mode=0o777)
    with open(f'data_resource/{timetag}/zheyi_roi.json', 'r', encoding='utf-8') as f:
        zheyi_roi_data = json.load(f)
    with open(f'data_resource/{timetag}/zheyi_slide.json', 'r', encoding='utf-8') as f:
        zheyi_slide = json.load(f)
    with open(f'data_resource/{timetag}/wxl_pos_slide.json', 'r', encoding='utf-8') as f:
        wxl_pos_slide = json.load(f)
    with open(f'data_resource/{timetag}/jfsw_pos_slide.json', 'r', encoding='utf-8') as f:
        jfsw_pos_slide = json.load(f)

    zheyi_pos_slide = [*zheyi_roi_data, *zheyi_slide, *wxl_pos_slide]
    for all_json_datas,patches_jsonname in zip([zheyi_pos_slide, jfsw_pos_slide], ['patches_in_RoI_pure', 'patches_in_RoI_jfsw']):

        # gene_patch_jsonlist(0, all_json_datas, npz_mask_save_dir, patch_npz_save_dir)

        cpu_num = 8
        set_split = np.array_split(range(len(all_json_datas)), cpu_num)
        print(f"Number of cores: {cpu_num}, set number of per core: {len(set_split[0])}")
        multiprocessing.set_start_method('spawn', force=True)
        workers = Pool(processes=cpu_num)
        processes = []
        for proc_id, set_group in enumerate(set_split):
            process_group = [all_json_datas[i] for i in set_group]
            p = workers.apply_async(gene_patch_jsonlist, (proc_id, process_group, npz_mask_save_dir, patch_npz_save_dir))
            processes.append(p)
        patchItems = []
        for p in processes:
            results = p.get()
            patchItems.extend(results)
        workers.close()
        workers.join()
        with open(f'{json_save_dir}/{patches_jsonname}.json', 'w', encoding='utf-8') as f:
            json.dump(patchItems, f, ensure_ascii=False)

        for tag in ['neg', 'partial_pos', 'total_pos']:
            os.makedirs(f'{img_save_dir}/{tag}', exist_ok=True, mode=0o777)
        multiprocessing.set_start_method('spawn', force=True)
        cut_patch_imgs(img_save_dir)

        statistic_imgs()
 
'''
WINDOW_SIZE = 1600, STRIDE = 1550:
['neg', 'total_pos', 'partial_pos']: [19203, 11616, 2568], [0, 0, 10666] ([19203, 11616, 13234])

WINDOW_SIZE = 1000, STRIDE = 950:
['neg', 'total_pos', 'partial_pos']: [56186, 17051, 2507], [0, 0, 10732] ([56186, 17051, 13239])

WINDOW_SIZE = 700, STRIDE = 650:
['neg', 'total_pos', 'partial_pos']: [70183, 15448, 8382], [0, 0, 29666] ([70183, 15448, 38048])

WINDOW_SIZE = 512, STRIDE = 450:
['neg', 'total_pos', 'partial_pos']: [138951, 20250, 10071], [0, 0, 37348] ([138951, 20250, 47419])
'''
