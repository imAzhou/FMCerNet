import json
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import matplotlib.pyplot as plt
import os
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from cerwsi.utils import KFBSlide,random_cut_square,calc_relative_coord,is_bbox_inside
import random
from scipy import sparse


test_patientId = [
    'ZY_ONLINE_1_45', 'ZY_ONLINE_1_196',    # 0409 Slide，0422 Slide
    'JFSW_2_1486', 'JFSW_2_152',     # 空 RoI，有标注 RoI
    'JFSW_1_2', 'JFSW_2_133', 'JFSW_2_2111', 'WXL_1_26'     # partial_pos slide
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

def vis_sample(image, roi_masks, roi_item, filename):
    plt.figure(figsize=(20, 20))
    plt.imshow(image)

    rx1,ry1,rx2,ry2 = roi_item['region']
    for child in roi_item['children']:
        bx1,by1,bx2,by2 = child['region']
        box = [bx1-rx1,by1-ry1,bx2-rx1,by2-ry1]
        show_box(box, plt.gca())
    
    annids = np.unique(roi_masks)
    for annid_idx in annids[1:]:   # 第一个是 0
        mask = roi_masks == annid_idx
        show_mask(mask, plt.gca(), random_color=True)
        
    plt.axis('off')
    plt.tight_layout()
    os.makedirs('statistic_results/0511/sam2_infer_RoI', exist_ok=True, mode=0o777)
    plt.savefig(f'statistic_results/0511/sam2_infer_RoI/{filename}')
    plt.close()

def gene_roi_lesion_mask(all_json_datas,npz_mask_save_dir):
    sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    device = torch.device("cuda:1")
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    for idx, item in enumerate(tqdm(all_json_datas, ncols=80)):
        # if idx < 422:
        #     continue
        # if item['patientId'] not in test_patientId:
        #     continue
        
        if item['media_type'] == 'roi':
            roi_img = Image.open(item['source_path'])
        elif item['media_type'] == 'slide':
            slide = KFBSlide(item['source_path'])

        for RoIItem in item['annotations']:
            purename = item['patientId'] + f'_{str(RoIItem["annid"])}'
            if os.path.exists(f'{npz_mask_save_dir}/{purename}.npz'):
                continue

            rx1,ry1,rx2,ry2 = RoIItem['region']
            rw,rh = int(rx2-rx1+0.5), int(ry2-ry1+0.5)
            roi_mask = np.zeros((rh,rw), dtype=np.int16)
            annid_list = [child['annid'] for child in RoIItem['children']]
            [child.update({'inst_infer': False}) for child in RoIItem['children']]
            for annitem in RoIItem['children']:
                if annitem['inst_infer']:
                    continue

                annx1,anny1,annx2,anny2 = annitem['region']
                annw, annh = int(annx2-annx1+0.5), int(anny2-anny1+0.5)
                maxlen = max(annw, annh)
                sq_size = max(maxlen, 1024)
                square_x1,square_y1 = random_cut_square((annx1,anny1,annw,annh), sq_size)
                square_x2,square_y2 = square_x1+sq_size,square_y1+sq_size
                square_x1,square_y1 = max(0,square_x1),max(0,square_y1)
                square_x2,square_y2 = min(square_x2,rx2),min(square_y2,ry2)
                
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
                        bbox_coord = calc_relative_coord(parent_patch_coords, annitem['region'])
                        if bbox_coord is not None:
                            input_boxes.append(bbox_coord)
                            input_boxes_annid.append(annitem['annid'])
                            annitem['inst_infer'] = True

                image = np.array(cropped.convert("RGB"))
                predictor.set_image(image)
                if len(input_boxes) == 0:
                    print('len(input_boxes) == 0')
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
                bool_masks = masks.astype(bool)
                num_masks = len(bool_masks)
                is_fully_contained = [False] * num_masks
                # 检查每个 mask 是否被其它任意一个完全包含
                for i in range(num_masks):
                    for j in range(num_masks):
                        if i == j:
                            continue
                        # 如果 mask i 被 mask j 完全包含（即 i 为 True 的地方 j 也为 True）
                        if np.all(bool_masks[i][bool_masks[i]] <= bool_masks[j][bool_masks[i]]):
                            is_fully_contained[i] = True
                            break  # 一旦发现被包含即可跳过

                # 遍历未被完全包裹的 mask
                for i, (mask, annid) in enumerate(zip(masks, input_boxes_annid)):
                    if is_fully_contained[i]:
                        continue  # 跳过被完全包裹的 mask
                    ys, xs = np.where(mask)     # 获取当前 mask 为 True 的位置坐标
                    annid_idx = annid_list.index(annid) + 1     # 索引从 1 开始，0 是默认的背景
                    shift_y, shift_x = int(square_y1 - ry1), int(square_x1 - rx1)
                    roi_mask[ys + shift_y, xs + shift_x] = annid_idx
            
            for annitem in RoIItem['children']:
                if not annitem['inst_infer']:
                    print(f'ERROR! Retain ann not infer: {purename}')
            if len(RoIItem['children']) > 0:
                sparse_mask = sparse.coo_matrix(roi_mask)  # 只保存非零元素的位置和值
                np.savez_compressed(f"{npz_mask_save_dir}/{purename}.npz",
                    data=sparse_mask.data,
                    row=sparse_mask.row,
                    col=sparse_mask.col,
                    shape=roi_mask.shape)
        

def test_roi_lesion_mask(all_json_datas,npz_mask_save_dir):
    for idx, item in enumerate(tqdm(all_json_datas, ncols=80)):
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
            if len(RoIItem['children']) > 50:
                continue
            rx1,ry1,rx2,ry2 = RoIItem['region']
            rw,rh = int(rx2-rx1+0.5), int(ry2-ry1+0.5)
            purename = item['patientId'] + f'_{str(RoIItem["annid"])}'
            if len(RoIItem['children']) > 0:
                loader = np.load(f"{npz_mask_save_dir}/{purename}.npz")
                sparse_mask = sparse.coo_matrix((loader['data'], (loader['row'], loader['col'])), shape=loader['shape'])
                roi_mask = sparse_mask.toarray().astype(np.int16)
            else:
                roi_mask = np.zeros((rh,rw), dtype=np.int16)
            
            SIZE_THR = 4000
            if rw > SIZE_THR or rh > SIZE_THR:
                start_x1,start_y1 = random_cut_square((0,0,rw,rh), SIZE_THR)
                start_x1,start_y1 = max(start_x1,0), max(start_y1,0)
                start_x2,start_y2 = min(start_x1+SIZE_THR,rw), min(start_y1+SIZE_THR,rh)
                roi_mask = roi_mask[start_y1:start_y2, start_x1:start_x2]
                RoIItem['region'] = [start_x1+rx1, start_y1+ry1, start_x2+rx1, start_y2+ry1]
                RoIItem['children'] = [i for i in RoIItem['children'] if is_bbox_inside(i['region'],RoIItem['region'],tolerance=10)]

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


if __name__ == "__main__":
    with open('data_resource/0511/ann_jsons/zheyi_roi.json', 'r', encoding='utf-8') as f:
        zheyi_roi_data = json.load(f)   # 996
    with open('data_resource/0511/ann_jsons/zheyi_slide.json', 'r', encoding='utf-8') as f:
        zheyi_slide = json.load(f)  # 60
    with open('data_resource/0511/ann_jsons/partial_pos_slide.json', 'r', encoding='utf-8') as f:
        partial_pos_slide = json.load(f)    # 1176

    all_json_datas = [*zheyi_roi_data, *zheyi_slide, *partial_pos_slide]
    # all_json_datas = [*zheyi_roi_data, *zheyi_slide]
    npz_mask_save_dir = 'data_resource/0511/roi_inst_mask'
    os.makedirs(npz_mask_save_dir, exist_ok=True, mode=0o777)
    gene_roi_lesion_mask(all_json_datas, npz_mask_save_dir)
    # test_roi_lesion_mask(all_json_datas, npz_mask_save_dir)

