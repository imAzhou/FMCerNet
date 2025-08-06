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
from cerwsi.utils import KFBSlide,random_cut_square,calc_relative_coord,is_bbox_inside,set_seed,generate_cut_regions
from scipy import sparse
from collections import defaultdict
from pycocotools.coco import COCO

device = torch.device("cuda:0")
POSITIVE_CLASS = ['AGC', 'ASC-US','LSIL', 'ASC-H', 'HSIL', 'SCC']
clsname_map = {
    'ascus': 'ASC-US',
    'lsil': 'LSIL',
    'asch': 'ASC-H',
    'hsil': 'HSIL',
    'scc': 'SCC',
    'agc': 'AGC',
    'trichomonas': 'NILM',
    'candida': 'NILM',
    'flora': 'NILM',
    'herps': 'NILM',
    'actinomyces': 'NILM',
}

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

def vis_instmask(image, roi_masks, bboxes, savepath):
    plt.figure(figsize=(20, 10))
    plt.imshow(image)

    # 1. 绘制 children 中的红色虚线框
    for bbox in bboxes:
        show_box(bbox, plt.gca(), color='red', linestyle='--')

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
    plt.savefig(savepath)
    plt.close()


def gene_roi_lesion_mask(all_json_datas, npz_mask_save_dir):
    sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    for idx, item in enumerate(tqdm(all_json_datas, ncols=80)):
        # if item['patientId'] not in test_patientId:
        #     continue
        imgpath = f"{root_dir}/{item['mode']}/{item['file_name']}"
        purename = item['file_name'].split('.')[0]
        roi_img = Image.open(imgpath)
        predictor.set_image(roi_img)
        rw,rh = roi_img.size
        roi_mask = np.zeros((rh,rw), dtype=np.int16)

        input_boxes, input_instid = [],[]
        for instid,anninfo in enumerate(item['annos']):
            x1,y1,w,h = anninfo['bbox']
            input_boxes.append([x1, y1, x1+w, y1+h])
            input_instid.append(instid+1)
        if len(input_boxes) > 0:
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
            empty_mask = 0
            for i, (mask, instid) in enumerate(zip(masks, input_instid)):
                if np.sum(mask) == 0:
                    empty_mask += 1
                ys, xs = np.where(mask)     # 获取当前 mask 为 True 的位置坐标
                roi_mask[ys, xs] = instid

            # vis_instmask(roi_img, roi_mask, input_boxes, f'{instmask_visdir}/{purename}.png')
            if empty_mask > 0:
                print(f'{purename} empty mask: {empty_mask}')
        sparse_mask = sparse.coo_matrix(roi_mask)  # 只保存非零元素的位置和值
        np.savez_compressed(f"{npz_mask_save_dir}/{purename}.npz",
            data=sparse_mask.data,
            row=sparse_mask.row,
            col=sparse_mask.col,
            shape=roi_mask.shape)

def visual_roi_mask(imgpath, annos, npz_mask_save_dir):
    image = Image.open(imgpath)
    purename = os.path.basename(imgpath).split('.')[0]
    loader = np.load(f"{npz_mask_save_dir}/{purename}.npz")
    sparse_mask = sparse.coo_matrix((loader['data'], (loader['row'], loader['col'])), shape=loader['shape'])
    roi_masks = sparse_mask.toarray().astype(np.int16)
    
    plt.figure(figsize=(20, 20))
    plt.imshow(image)


    # 1. 绘制 children 中的红色虚线框
    for anninfo in annos:
        x,y,w,h = anninfo['bbox']
        box = [x, y, x+w, y+h]
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
    plt.savefig(f'statistic_results/CDetector/sam2_infer_RoI/{purename}.png')
    plt.close()    

if __name__ == "__main__":
    root_dir = 'data_resource/ComparisonDetectorDataset'
    npz_mask_save_dir = f'{root_dir}/roi_inst_mask'
    os.makedirs(npz_mask_save_dir, exist_ok=True, mode=0o777)
    instmask_visdir = 'statistic_results/CDetector/sam2_infer_RoI'
    os.makedirs(instmask_visdir, exist_ok=True, mode=0o777)
    
    # all_json_datas = []
    # for mode in ['train','test']:
    #     jsonfile = f'{root_dir}/{mode}.json'
    #     with open(jsonfile, 'r', encoding='utf-8') as f:
    #         json_data = json.load(f)
    #     coco = COCO(jsonfile)
    #     for imgitem in json_data['images']:
    #         imgitem['mode'] = mode
    #         annids = coco.getAnnIds([imgitem['id']])
    #         annos = coco.loadAnns(annids)
    #         imgitem['annos'] = []
    #         for ann in annos:
    #             catinfo = coco.loadCats([ann['category_id']])[0]
    #             clsname = clsname_map[catinfo['name']]
    #             if ann['bbox'][2]>5 and ann['bbox'][3]>5 and clsname in POSITIVE_CLASS:
    #                 imgitem['annos'].append(ann)
    #         all_json_datas.append(imgitem)
    # gene_roi_lesion_mask(all_json_datas, npz_mask_save_dir)

    imgpath = f'{root_dir}/test/05838.bmp'
    jsonfile = f'{root_dir}/test.json'
    coco = COCO(jsonfile)
    annids = coco.getAnnIds(['05838.bmp'])
    annos = coco.loadAnns(annids)
    visual_roi_mask(imgpath, annos, npz_mask_save_dir)
