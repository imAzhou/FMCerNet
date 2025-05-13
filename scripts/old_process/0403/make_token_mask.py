from segment_anything import sam_model_registry, SamPredictor
import json
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
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
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    for box in input_boxes:
        show_box(box.cpu().numpy(), plt.gca())
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'statistic_results/0403/sam_output_mask/{filename}')
    plt.close()

def make_token_mask():
    path_prefix = 'data_resource/0403/images/Pos'
    os.makedirs('statistic_results/0403/sam_output_mask', exist_ok=True)
    os.makedirs('data_resource/0403/mask', exist_ok=True)

    sam_checkpoint = "/nfs5/zly/codes/segment-anything/checkpoints/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda:0"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)

    for mode in ['train', 'val']:
        with open(f'data_resource/0403/annofiles/{mode}_posslide_patches_filtered.json', 'r') as f:
            kfb_list = json.load(f)
        
        for idx, kfbinfo in enumerate(tqdm(kfb_list, ncols=80)):
            patch_list = kfbinfo['patch_list']
            for patchinfo in patch_list:
                # if random.random() > 0.01:
                #     continue
                imgpath = f'{path_prefix}/{patchinfo["filename"]}'
                image = cv2.imread(imgpath)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                predictor.set_image(image)
                input_boxes = torch.tensor(patchinfo['bboxes'], device=predictor.device)
                transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
                # masks: (k,1,h,w)
                masks, _, _ = predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    multimask_output=False,
                )

                h, w = image.shape[:2]
                gt_mask = np.zeros((h, w), dtype=int)
                for mask, clsname in zip(masks, patchinfo['clsnames']):
                    if clsname in NEGATIVE_CLASS:
                        clsid = 1
                    else:
                        clsid = classes.index(RECORD_CLASS[clsname]) + 1
                    forground_mask = mask[0].detach().cpu().numpy()
                    gt_mask[forground_mask] = clsid
                purename = patchinfo["filename"].split('.')[0]
                # 获取非零元素的索引和值
                nonzero_indices = np.array(np.nonzero(gt_mask))  # 非零元素索引
                nonzero_values = gt_mask[nonzero_indices[0], nonzero_indices[1]]  # 非零元素值
                # 保存索引和值
                np.savez_compressed(f'data_resource/0403/mask/{purename}.npz', indices=nonzero_indices, values=nonzero_values, shape=gt_mask.shape)

                # 读取并还原 gt_mask
                # data = np.load(f'data_resource/0403/mask/{purename}.npz')
                # nonzero_indices = data['indices']
                # nonzero_values = data['values']
                # shape = tuple(data['shape'])
                # restored_gt_mask = np.zeros(shape, dtype=int)
                # restored_gt_mask[nonzero_indices[0], nonzero_indices[1]] = nonzero_values

if __name__ == '__main__':
    pass