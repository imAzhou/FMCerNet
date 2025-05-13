# from sam2.build_sam import build_sam2
# from sam2.sam2_image_predictor import SAM2ImagePredictor
import json
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import random
import os

NEGATIVE_CLASS = ['NILM', 'GEC']
POSITIVE_CLASS = ['ASC-US', 'LSIL', 'ASC-H', 'HSIL', 'SCC', 'AGC-NOS', 'AGC', 'AGC-N', 'AGC-FN']
classes = ['NILM', 'AGC', 'ASC-US', 'LSIL', 'ASC-H', 'HSIL']
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

def vis_sample(image,masks,input_boxes,filename):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
        show_mask(mask.squeeze(0), plt.gca(), random_color=True)
    for box in input_boxes:
        show_box(box, plt.gca())
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'statistic_results/0416/sam2_output_mask/{filename}')
    plt.close()

def make_token_mask():
    # path_prefix = '/c22073/zly/datasets/CervicalDatasets/LCerScanv4/images'
    # os.makedirs('statistic_results/0416/sam2_output_mask', exist_ok=True)
    # mask_save_dir = '/c22073/zly/datasets/CervicalDatasets/LCerScanv4/mask'
    # os.makedirs(mask_save_dir, exist_ok=True)

    sam2_checkpoint = "/c22073/zly/codes/sam2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    # device = torch.device("cuda:7")
    # torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # if torch.cuda.get_device_properties(0).major >= 8:
    #     torch.backends.cuda.matmul.allow_tf32 = True
    #     torch.backends.cudnn.allow_tf32 = True
    # sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    # predictor = SAM2ImagePredictor(sam2_model)

    for mode in ['train', 'val']:
        with open(f'data_resource/0416/annofiles/{mode}_partial_pos.json', 'r') as f:
            patch_list = (json.load(f))['']
        
        for patchinfo in tqdm(patch_list, ncols=80):
            # if random.random() > 0.01:
            #     continue
            if patchinfo['diagnose'] == 0:
                continue
            if patchinfo["filename"] == 'JFSW_1_94_361.png':
                print()
            imgpath = f'{path_prefix}/{patchinfo["prefix"]}/{patchinfo["filename"]}'
            image = Image.open(imgpath)
            image = np.array(image.convert("RGB"))
            h, w = image.shape[:2]

            predictor.set_image(image)
            input_boxes = np.array(patchinfo['bboxes'])
            # masks: (k,1,h,w)
            masks, scores, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )
            if len(masks.shape) == 3:
                masks = masks[None,:]

            # vis_sample(image,masks,input_boxes,patchinfo["filename"])
            if patchinfo['prefix'] == 'total_pos':
                gt_mask = np.ones((h, w), dtype=int)
            else:
                gt_mask = np.zeros((h, w), dtype=int)   # 0: 未知，1: 阴性，>1: 阳性
            
            for mask, clsname in zip(masks, patchinfo['clsnames']):
                if clsname in NEGATIVE_CLASS:
                    clsid = 1
                else:
                    clsid = classes.index(RECORD_CLASS[clsname]) + 1
                forground_mask = mask[0] > 0
                gt_mask[forground_mask] = clsid
            purename = patchinfo["filename"].split('.')[0]
            # 获取非零元素的索引和值
            nonzero_indices = np.array(np.nonzero(gt_mask))  # 非零元素索引
            nonzero_values = gt_mask[nonzero_indices[0], nonzero_indices[1]]  # 非零元素值
            # 保存索引和值
            np.savez_compressed(f'{mask_save_dir}/{purename}.npz', indices=nonzero_indices, values=nonzero_values, shape=gt_mask.shape)

            # 读取并还原 gt_mask
            # data = np.load(f'data_resource/0403/mask/{purename}.npz')
            # nonzero_indices = data['indices']
            # nonzero_values = data['values']
            # shape = tuple(data['shape'])
            # restored_gt_mask = np.zeros(shape, dtype=int)
            # restored_gt_mask[nonzero_indices[0], nonzero_indices[1]] = nonzero_values

if __name__ == '__main__':
    make_token_mask()