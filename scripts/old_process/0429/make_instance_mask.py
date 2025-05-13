from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import json
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from PIL import Image
import os

NEGATIVE_CLASS = ['NILM', 'GEC']
POSITIVE_CLASS = ['ASC-US', 'LSIL', 'ASC-H', 'HSIL', 'SCC', 'AGC-NOS', 'AGC', 'AGC-N', 'AGC-FN']
classes = ['NILM', 'AGC', 'ASC-US', 'LSIL', 'ASC-H', 'HSIL']
# 类别映射关系
RECORD_CLASS = {
    'NILM': 'NILM',
    'GEC': 'NILM',
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
    os.makedirs('statistic_results/0429/sam2_output_mask', exist_ok=True)
    plt.savefig(f'statistic_results/0429/sam2_output_mask/{filename}')
    plt.close()

def make_instance_mask():
    path_prefix = '/c22073/zly/datasets/CervicalDatasets/LCerScanv1_512/images'
    mask_save_dir = '/c22073/zly/datasets/CervicalDatasets/LCerScanv1_512/instance_mask'
    os.makedirs(mask_save_dir, exist_ok=True)

    sam2_checkpoint = "/c22073/zly/codes/sam2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    device = torch.device("cuda:7")
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    for mode in ['train', 'val']:
        with open(f'/c22073/zly/datasets/CervicalDatasets/LCerScanv1_512/annofiles/{mode}.json', 'r') as f:
            patch_list = json.load(f)
        
        for idx, patchinfo in enumerate(tqdm(patch_list, ncols=80)):
            if patchinfo['diagnose'] == 0:
                continue
            # if idx > 20:
            #     break
            imgpath = f'{path_prefix}/{patchinfo["prefix"]}/{patchinfo["filename"]}'
            image = Image.open(imgpath)
            image = np.array(image.convert("RGB"))

            predictor.set_image(image)
            input_boxes,input_boxes_labelid = [],[]
            for clsname,bbox in zip(patchinfo['clsnames'], patchinfo['bboxes']):
                clsid = classes.index(RECORD_CLASS[clsname])
                if clsid > 0:
                    input_boxes.append(bbox)
                    input_boxes_labelid.append(clsid)
            input_boxes = np.array(input_boxes)
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

            purename = patchinfo["filename"].split('.')[0]
            masks_to_save = masks.squeeze(1)  # (n, h, w)
            np.savez_compressed(f'{mask_save_dir}/{purename}.npz', masks=masks_to_save, labels=input_boxes_labelid)

            # data = np.load(f'{mask_save_dir}/{purename}.npz')
            # masks = data['masks']      # (n, h, w)
            # labels = data['labels']    # (n,)
            # print()
            

if __name__ == '__main__':
    make_instance_mask()