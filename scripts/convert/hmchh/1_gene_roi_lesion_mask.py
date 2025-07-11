import json
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import matplotlib.pyplot as plt
import os
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from cerwsi.utils import KFBSlide,random_cut_square,calc_relative_coord,is_bbox_inside,set_seed
from scipy import sparse
from collections import defaultdict
import multiprocessing
from multiprocessing import Pool

test_patientId = [
    '1657bj008',
    # 'ZY_ONLINE_1_45', 'JFSW_2_200',    # 0409 Slide，0422 Slide
    # 'JFSW_2_1486', 'JFSW_2_152',     # 空 RoI，有标注 RoI
    # 'JFSW_2_2111', 'JFSW_2_133', 'WXL_1_26'     # partial_pos slide
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
        plt.contour(mask, levels=[0.5], colors='lime', linewidths=2)
        
    plt.axis('off')
    plt.tight_layout()
    os.makedirs('statistic_results/HMCHH/sam2_infer_RoI', exist_ok=True, mode=0o777)
    plt.savefig(f'statistic_results/HMCHH/sam2_infer_RoI/{filename}')
    plt.close()

def gene_roi_lesion_mask(proc_id, all_json_datas, npz_mask_save_dir):
    sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    device = torch.device("cuda:1")
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    empty_mask = defaultdict(int)
    for idx, item in enumerate(tqdm(all_json_datas, ncols=80)):
        # if item['patientId'] not in test_patientId:
        #     continue

        roi_img = Image.open(item['source_path'])

        for RoIItem in item['annotations']:
            purename = item['patientId'] + f'_{str(RoIItem["annid"])}'
            if os.path.exists(f'{npz_mask_save_dir}/{purename}.npz'):
                continue

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
                
                parent_patch_coords = [square_x1,square_y1,square_x2,square_y2]
                cropped = roi_img.crop(parent_patch_coords)
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

                for i, (mask, annid) in enumerate(zip(masks, input_boxes_annid)):
                    if np.sum(mask) == 0:
                        empty_mask[purename] += 1
                    ys, xs = np.where(mask)     # 获取当前 mask 为 True 的位置坐标
                    annid_idx = annid_list.index(annid) + 1     # 索引从 1 开始，0 是默认的背景
                    shift_y, shift_x = int(square_y1 - ry1), int(square_x1 - rx1)
                    roi_mask[ys + shift_y, xs + shift_x] = annid_idx
            
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

        if item['patientId'] not in test_patientId:
            continue
        roi_img = Image.open(item['source_path'])

        for RoIItem in item['annotations']:
            rx1,ry1,rx2,ry2 = RoIItem['region']
            rw,rh = int(rx2-rx1+0.5), int(ry2-ry1+0.5)
            purename = item['patientId'] + f'_{str(RoIItem["annid"])}'
            loader = np.load(f"{npz_mask_save_dir}/{purename}.npz")
            sparse_mask = sparse.coo_matrix((loader['data'], (loader['row'], loader['col'])), shape=loader['shape'])
            roi_mask = sparse_mask.toarray().astype(np.int16)                
            print(f'\nDrawing {purename}.png, (width,height) is {roi_img.size}')
            vis_sample(roi_img, roi_mask, RoIItem, f'{purename}.png')


def test_file_exist(all_json_datas,npz_mask_save_dir):
    for idx, item in enumerate(tqdm(all_json_datas, ncols=80)):
        if idx != 1585:
            continue
        for RoIItem in item['annotations']:
            purename = item['patientId'] + f'_{str(RoIItem["annid"])}'
            flag = False
            if len(RoIItem['children']) > 0:
                loader = np.load(f"{npz_mask_save_dir}/{purename}.npz")
                sparse_mask = sparse.coo_matrix((loader['data'], (loader['row'], loader['col'])), shape=loader['shape'])
                roi_mask = sparse_mask.toarray().astype(np.int16)
                # unique_idx = np.unique(roi_mask)
                if np.sum(roi_mask) == 0:
                    flag = True
            if flag:
                print(f'Error in {purename}')


if __name__ == "__main__":
    set_seed(666)
    with open('data_resource/HMCHH/annofiles_roi/unify_ann.json', 'r', encoding='utf-8') as f:
        roi_data = json.load(f)

    npz_mask_save_dir = 'data_resource/HMCHH/roi_inst_mask'
    os.makedirs(npz_mask_save_dir, exist_ok=True, mode=0o777)

    # cpu_num = 8
    # set_split = np.array_split(range(len(roi_data)), cpu_num)
    # print(f"Number of cores: {cpu_num}, set number of per core: {len(set_split[0])}")
    # multiprocessing.set_start_method('spawn', force=True)
    # workers = Pool(processes=cpu_num)
    # processes = []
    # for proc_id, set_group in enumerate(set_split):
    #     process_group = [roi_data[i] for i in set_group]
    #     p = workers.apply_async(gene_roi_lesion_mask, (proc_id, process_group, npz_mask_save_dir))
    #     processes.append(p)
    # for p in processes:
    #     p.get()
    # workers.close()
    # workers.join()
    
    test_roi_lesion_mask(roi_data, npz_mask_save_dir)
