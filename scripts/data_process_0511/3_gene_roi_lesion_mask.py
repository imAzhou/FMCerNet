import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import torch
import matplotlib.pyplot as plt
import os
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from cerwsi.utils import KFBSlide,random_cut_square,calc_relative_coord,is_bbox_inside,set_seed
from scipy import sparse
import glob
from collections import defaultdict
import multiprocessing
from multiprocessing import Pool

test_patientId = [
    'ZY_ONLINE_1_45', 'ZY_ONLINE_1_21',    # 0409 Slide，0422 Slide
    'JFSW_2_1486', 'JFSW_2_152', 'JFSW_2_239',     # 空 RoI，有标注 RoI
    'WXL_1_26',    # partial_pos slide in pure_train
    'JFSW_2_2111', 'JFSW_2_2',     # partial_pos slide in jfsw_train
    'JFSW_2_302',
]

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax, color='green', linestyle='-'):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=color, linestyle=linestyle,
                               facecolor=(0, 0, 0, 0), lw=2))

def vis_sample(image, roi_masks, roi_item, filename):
    plt.figure(figsize=(20, 20))
    plt.imshow(image)

    rx1, ry1, rx2, ry2 = roi_item['region']

    # 1. 绘制 children 中的红色虚线框
    for child in roi_item['children']:
        bx1, by1, bx2, by2 = child['region']
        box = [bx1 - rx1, by1 - ry1, bx2 - rx1, by2 - ry1]
        show_box(box, plt.gca(), color='red', linestyle='--')

    # 2. 绘制 mask 和 mask 的外接框
    annids = np.unique(roi_masks)
    for annid_idx in annids[1:]:  # 第一个是 0
        mask = roi_masks == annid_idx
        # show_mask(mask, plt.gca(), random_color=True)
        plt.contour(mask, levels=[0.5], colors='lime', linewidths=2)

        # 外接框（绿色实线）
        yx = np.argwhere(mask)
        if yx.size > 0:
            y_min, x_min = yx.min(axis=0)
            y_max, x_max = yx.max(axis=0)
            show_box([x_min, y_min, x_max, y_max], plt.gca(), color='green', linestyle='-')

    plt.axis('off')
    plt.tight_layout()
    os.makedirs('statistic_results/0511/sam2_infer_RoI', exist_ok=True, mode=0o777)
    plt.savefig(f'statistic_results/0511/sam2_infer_RoI/{filename}')
    plt.close()

def postprocess_mask(masks, input_boxes, input_boxes_annid, sort_by_area=True):
    '''
    mask: ndarry (n, h, w)
    input_boxes: ndarry (n, 4)
    input_boxes_annid: list of ann item annid

    如果 input_boxes 中的某个 bbox 是被其他bbox包含，
    且两者对应的 mask 有交集(该bbox的mask有一半及以上在父box的mask中)，就把这个bbox丢弃
    '''

    bool_masks = masks.astype(bool)
    num_masks = len(bool_masks)
    to_remove = [False] * num_masks

    for i in range(num_masks):
        for j in range(num_masks):
            if i == j or to_remove[i]:
                continue

            if is_bbox_inside(input_boxes[i], input_boxes[j], 5):
                mask_i = bool_masks[i]
                mask_j = bool_masks[j]
                intersection = np.logical_and(mask_i, mask_j).sum()
                area_i = mask_i.sum()

                if area_i > 0 and intersection / area_i >= 0.3:
                    to_remove[i] = True
                    break

    keep_indices = [i for i, remove in enumerate(to_remove) if not remove]

    filtered_masks = [masks[i] for i in keep_indices]
    filtered_boxes = [input_boxes[i] for i in keep_indices]
    filtered_annids = [input_boxes_annid[i] for i in keep_indices]

    if sort_by_area:
        areas = [mask.astype(bool).sum() for mask in filtered_masks]
        sorted_idx = sorted(range(len(filtered_masks)), key=lambda i: -areas[i])
        filtered_masks = [filtered_masks[i] for i in sorted_idx]
        filtered_boxes = [filtered_boxes[i] for i in sorted_idx]
        filtered_annids = [filtered_annids[i] for i in sorted_idx]

    return np.array(filtered_masks), filtered_annids
        
def gene_roi_lesion_mask(proc_id, all_json_datas, npz_mask_save_dir):
    sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    device = torch.device("cuda:2")
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    empty_mask = defaultdict(int)
    for idx, item in enumerate(all_json_datas):
        # if item['patientId'] not in test_patientId:
        #     continue

        if item['media_type'] == 'roi':
            roi_img = Image.open(item['source_path'])
        elif item['media_type'] == 'slide':
            slide = KFBSlide(item['source_path'])

        for RoIItem in item['annotations']:
            purename = item['patientId'] + f'_{str(RoIItem["annid"])}'
            # if os.path.exists(f'{npz_mask_save_dir}/{purename}.npz'):
            #     continue

            rx1,ry1,rx2,ry2 = RoIItem['region']
            rw,rh = int(rx2-rx1+0.5), int(ry2-ry1+0.5)
            roi_mask = np.zeros((rh,rw), dtype=np.int16)
            # 按照bbox area 从大到小排序
            RoIItem['children'] = sorted(
                RoIItem['children'],
                key=lambda annitem: (annitem['region'][2] - annitem['region'][0]) * (annitem['region'][3] - annitem['region'][1]),
                reverse=True  # 从大到小排序
            )
            annid_list = [child['annid'] for child in RoIItem['children']]
            [child.update({'inst_infer': False}) for child in RoIItem['children']]
            for annitem in RoIItem['children']:
                if annitem['inst_infer']:
                    continue

                annx1,anny1,annx2,anny2 = annitem['region']
                annw, annh = int(annx2-annx1+0.5), int(anny2-anny1+0.5)
                sq_size = max(1024, max(annw, annh))
                square_x1,square_y1 = random_cut_square((annx1,anny1,annw,annh), sq_size)
                square_x2,square_y2 = square_x1+sq_size,square_y1+sq_size
                square_x1,square_y1 = max(0,square_x1),max(0,square_y1)
                if square_x2 > rx2:
                    square_x1 = square_x1 - (square_x2-rx2)
                    square_x2 = square_x1+sq_size
                if square_y2 > ry2:
                    square_y1 = square_y1 - (square_y2-ry2)
                    square_y2 = square_y1+sq_size
                
                if item['media_type'] == 'roi':
                    cropped = roi_img.crop((square_x1,square_y1,square_x2,square_y2))
                elif item['media_type'] == 'slide':
                    square_w,square_h = square_x2-square_x1, square_y2-square_y1
                    location, level, size = (square_x1,square_y1), 0, (square_w,square_h)
                    cropped = Image.fromarray(slide.read_region(location, level, size))
                
                parent_patch_coords = [square_x1,square_y1,square_x2,square_y2]
                input_boxes,input_boxes_annid = [],[]
                for annitem in RoIItem['children']:
                    if (not annitem['inst_infer']) and is_bbox_inside(annitem['region'],parent_patch_coords,tolerance=10):
                        intx1,inty1,intx2,inty2 = (np.array(annitem['region']).astype(int)).tolist()
                        bbox_coord = calc_relative_coord(parent_patch_coords, [intx1,inty1,intx2,inty2])
                        if bbox_coord is not None:
                            input_boxes.append(bbox_coord)
                            input_boxes_annid.append(annitem['annid'])
                            annitem['inst_infer'] = True
                
                if len(input_boxes) == 0:
                    print(f'purename {purename}: len(input_boxes) == 0')

                image = np.array(cropped.convert("RGB"))
                predictor.set_image(image)
                
                input_boxes = np.array(input_boxes)
                masks, scores, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_boxes,
                    multimask_output=False,
                )
                if len(masks.shape) == 3:
                    masks = masks[None,:]
                masks = masks.squeeze(1)  # (n, h, w)
                masks, input_boxes_annid = postprocess_mask(masks, input_boxes, input_boxes_annid,sort_by_area=False)

                for i, (mask, annid) in enumerate(zip(masks, input_boxes_annid)):
                    if np.sum(mask) == 0:
                        empty_mask[purename] += 1
                    ys, xs = np.where(mask)     # 获取当前 mask 为 True 的位置坐标
                    annid_idx = annid_list.index(annid) + 1     # 索引从 1 开始，0 是默认的背景
                    shift_y, shift_x = int(square_y1 - ry1), int(square_x1 - rx1)
                    roi_mask[ys + shift_y, xs + shift_x] = annid_idx
                
                # start_x1,start_y1,start_x2,start_y2 = square_x1 - rx1,square_y1 - ry1,square_x2 - rx1,square_y2 - ry1
                # roi_mask = roi_mask[start_y1:start_y2, start_x1:start_x2]
                # RoIItem['children'] = [i for i in RoIItem['children'] if i['annid'] in input_boxes_annid]
                # RoIItem['region'] = parent_patch_coords
                # vis_sample(image, roi_mask, RoIItem, f'{purename}.png')
            
            for annitem in RoIItem['children']:
                if not annitem['inst_infer']:
                    print(f'ERROR! Retain ann not infer: {purename}')
            if empty_mask[purename] > 0:
                print(f'{purename} empty mask: {empty_mask[purename]}')
                
            if len(RoIItem['children']) > 0:
                sparse_mask = sparse.coo_matrix(roi_mask)  # 只保存非零元素的位置和值
                np.savez_compressed(f"{npz_mask_save_dir}/{purename}.npz",
                    data=sparse_mask.data,
                    row=sparse_mask.row,
                    col=sparse_mask.col,
                    shape=roi_mask.shape)
        
        print(f'Core {proc_id} processed : {idx+1}/{len(all_json_datas)}.')

def test_roi_lesion_mask(all_json_datas,npz_mask_save_dir):
    for idx, item in enumerate(tqdm(all_json_datas, ncols=80)):
        # if idx != 1585:
        #     continue
        # if item['patientId'] != 'JFSW_2_107':
        #     continue
        if item['patientId'] not in test_patientId:
            continue
        if item['media_type'] == 'roi':
            roi_img = Image.open(item['source_path'])
        elif item['media_type'] == 'slide':
            slide = KFBSlide(item['source_path'])

        draw_cnt = 0
        for RoIItem in item['annotations']:
            if draw_cnt > 3:
                break
            # if len(RoIItem['children']) > 50:
            #     continue
            # if str(RoIItem['annid']) != '1457140814755':
            #     continue
            rx1,ry1,rx2,ry2 = RoIItem['region']
            rw,rh = int(rx2-rx1+0.5), int(ry2-ry1+0.5)
            purename = item['patientId'] + f'_{str(RoIItem["annid"])}'
            # if purename != 'JFSW_2_88_5205493336944':
            #     continue
            # if os.path.exists(f'statistic_results/0511/sam2_infer_RoI/{purename}.png'):
            #     continue
            if len(RoIItem['children']) == 0:
                continue
            loader = np.load(f"{npz_mask_save_dir}/{purename}.npz")
            sparse_mask = sparse.coo_matrix((loader['data'], (loader['row'], loader['col'])), shape=loader['shape'])
            roi_mask = sparse_mask.toarray().astype(np.int16)
            
            SIZE_THR = 2000 if len(RoIItem['children']) > 30 else 4000
            if rw > SIZE_THR or rh > SIZE_THR:
                start_x1,start_y1 = random_cut_square((0,0,rw,rh), SIZE_THR)
                start_x1,start_y1 = max(start_x1,0), max(start_y1,0)
                start_x2,start_y2 = min(start_x1+SIZE_THR,rw), min(start_y1+SIZE_THR,rh)
                RoIItem['region'] = [start_x1+rx1, start_y1+ry1, start_x2+rx1, start_y2+ry1]
                RoIItem['children'] = [i for i in RoIItem['children'] if is_bbox_inside(i['region'],RoIItem['region'],tolerance=10)]
                roi_mask = roi_mask[start_y1:start_y2, start_x1:start_x2]

                if item['media_type'] == 'roi':
                    sample_img = roi_img.crop([start_x1,start_y1,start_x2,start_y2])
                elif item['media_type'] == 'slide':
                    rx1,ry1,rx2,ry2 = RoIItem['region']
                    rw,rh = int(rx2-rx1+0.5), int(ry2-ry1+0.5)
                    location, level, size = (rx1,ry1), 0, (rw,rh)
                    sample_img = Image.fromarray(slide.read_region(location, level, size))
            else:
                if item['media_type'] == 'roi':
                    sample_img = roi_img
                elif item['media_type'] == 'slide':
                    location, level, size = (rx1,ry1), 0, (rw,rh)
                    sample_img = Image.fromarray(slide.read_region(location, level, size))

            print(f'\nDrawing {purename}.png, (width,height) is {sample_img.size}')
            vis_sample(sample_img, roi_mask, RoIItem, f'{purename}.png')
            draw_cnt += 1
            
def statistic_roi_anncnts(all_json_datas,npz_mask_save_dir):
    annos_cnt, annos_cnt_filterd = 0,0
    for idx, item in enumerate(tqdm(all_json_datas, ncols=80)):
        for RoIItem in item['annotations']:
            
            rx1,ry1,rx2,ry2 = RoIItem['region']
            rw,rh = int(rx2-rx1+0.5), int(ry2-ry1+0.5)
            purename = item['patientId'] + f'_{str(RoIItem["annid"])}'
            # if purename != 'JFSW_2_88_5205493336944':
            #     continue
            # if os.path.exists(f'statistic_results/0511/sam2_infer_RoI/{purename}.png'):
            #     continue
            if len(RoIItem['children']) == 0:
                continue
            loader = np.load(f"{npz_mask_save_dir}/{purename}.npz")
            sparse_mask = sparse.coo_matrix((loader['data'], (loader['row'], loader['col'])), shape=loader['shape'])
            roi_mask = sparse_mask.toarray().astype(np.int16)
            annids = np.unique(roi_mask)
            annos_cnt_filterd += len(annids[1:])
            annos_cnt += len(RoIItem['children'])
    
    print(f'{annos_cnt_filterd}/{annos_cnt}')


def test_file_exist(all_json_datas,npz_mask_save_dir):

    keep_names = []
    minw,minh = float('inf'), float('inf')
    for idx, item in enumerate(tqdm(all_json_datas, ncols=80)):
        for RoIItem in item['annotations']:
            purename = item['patientId'] + f'_{str(RoIItem["annid"])}'
            rx1,ry1,rx2,ry2 = RoIItem['region']
            rw,rh = rx2-rx1, ry2-ry1
            if rw < minw:
                minw = rw
            if rh < minh:
                minh = rh
            if len(RoIItem['children']) > 0 and not os.path.exists(f"{npz_mask_save_dir}/{purename}.npz"):
                print(f"Not exist: {purename}.npz")
    print(f'minw:{minw}, minh:{minh}')
    # for existname in os.listdir(npz_mask_save_dir):
    #     if existname not in keep_names and os.path.exists(f"{npz_mask_save_dir}/{purename}.npz"):
    #         os.remove(f"{npz_mask_save_dir}/{purename}.npz")

def clear_npz(all_json_datas,npz_mask_save_dir):

    keep_filename = []
    for idx, item in enumerate(tqdm(all_json_datas, ncols=80)):
        for RoIItem in item['annotations']:
            if len(RoIItem['children']) == 0:
                continue
            purename = item['patientId'] + f'_{str(RoIItem["annid"])}.npz'
            keep_filename.append(purename)

    exists_npzpath = glob.glob(f'{npz_mask_save_dir}/*.npz')
    print(f'keep_filename nums: {len(keep_filename)}')
    print(f'exists_imgpath nums: {len(exists_npzpath)}')

    for npzpath in tqdm(exists_npzpath, ncols=80):
        filename = os.path.basename(npzpath)
        if filename not in keep_filename:
            os.remove(npzpath)

if __name__ == "__main__":
    set_seed(666)
    with open('data_resource/0511/zheyi_roi.json', 'r', encoding='utf-8') as f:
        zheyi_roi_data = json.load(f)   # 1287
    with open('data_resource/0511/zheyi_slide.json', 'r', encoding='utf-8') as f:
        zheyi_slide = json.load(f)  # 60
    with open('data_resource/0511/wxl_pos_slide.json', 'r', encoding='utf-8') as f:
        wxl_pos_slide = json.load(f)    # 37
    with open('data_resource/0511/jfsw_pos_slide.json', 'r', encoding='utf-8') as f:
        jfsw_pos_slide = json.load(f)    # 308

    all_json_datas = [*zheyi_roi_data, *zheyi_slide, *wxl_pos_slide, *jfsw_pos_slide]
    npz_mask_save_dir = 'data_resource/0511/roi_inst_mask'
    os.makedirs(npz_mask_save_dir, exist_ok=True, mode=0o777)
    
    # cpu_num = 8
    # set_split = np.array_split(range(len(all_json_datas)), cpu_num)
    # print(f"Number of cores: {cpu_num}, set number of per core: {len(set_split[0])}")
    # multiprocessing.set_start_method('spawn', force=True)
    # workers = Pool(processes=cpu_num)
    # processes = []
    # for proc_id, set_group in enumerate(set_split):
    #     process_group = [all_json_datas[i] for i in set_group]
    #     p = workers.apply_async(gene_roi_lesion_mask, (proc_id, process_group, npz_mask_save_dir))
    #     processes.append(p)
    # for p in processes:
    #     p.get()
    # workers.close()
    # workers.join()
    
    # test_roi_lesion_mask(all_json_datas, npz_mask_save_dir)
    # test_file_exist(all_json_datas,npz_mask_save_dir)
    # clear_npz(all_json_datas,npz_mask_save_dir)
    statistic_roi_anncnts(all_json_datas, npz_mask_save_dir)    # 39447/40844
    