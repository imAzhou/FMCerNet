from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

import json
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from PIL import Image
import os
import glob
import os
import cv2

np.random.seed(3)

def show_anns_on_image(image, anns, borders=True):
    if len(anns) == 0:
        return image
    
    img_np = np.array(image).astype(np.float32) / 255.0  # [H, W, 4]
    sorted_anns = sorted(anns, key=lambda x: x['area'], reverse=True)

    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.2]])
        # img_np[m] = img_np[m] * (1 - color_mask[3]) + color_mask  # alpha blending

        if borders:
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = [cv2.approxPolyDP(c, epsilon=0.01, closed=True) for c in contours]
            color_boundary = tuple((color_mask[:3] * 255).astype(int).tolist())
            # OpenCV expects image in BGR format and uint8
            overlay = (img_np[:, :, :3] * 255).astype(np.uint8).copy()
            cv2.drawContours(overlay, contours, -1, color_boundary, thickness=2)
            img_np[:, :, :3] = overlay.astype(np.float32) / 255.0

    img_np = (img_np * 255).astype(np.uint8)
    return Image.fromarray(img_np, mode="RGBA")
    

def get_sam_autopredictor(sam_type):
    device = torch.device("cuda:2")

    if sam_type == 'sam':
        sam_checkpoint = "checkpoints/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,  # Requires open-cv to run post-processing
        )

    elif sam_type == 'sam2':
        sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
        mask_generator = SAM2AutomaticMaskGenerator(
            model=sam2,
            points_per_side=64,
            points_per_batch=128,
            pred_iou_thresh=0.7,
            stability_score_thresh=0.92,
            stability_score_offset=0.7,
            crop_n_layers=1,
            box_nms_thresh=0.7,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100.0,
            use_m2m=True,
        )
    
    return mask_generator

def everything_infer():
    root_dir = 'data_resource/cervical_cell_seg'
    image_dir = f'{root_dir}/images'
    sam_type = 'sam'
    mask_save_dir = f'{root_dir}/image_use_{sam_type}'
    json_save_dir = f'{root_dir}/resultjson_use_{sam_type}'
    os.makedirs(mask_save_dir, exist_ok=True)
    os.makedirs(json_save_dir, exist_ok=True)

    mask_generator = get_sam_autopredictor(sam_type)
    
    for imgpath in tqdm(glob.glob(f'{image_dir}/*.png'), ncols=80):
        read_image = Image.open(imgpath)
        filename = os.path.basename(imgpath)
        purename = filename.replace('.png', '')
        image = np.array(read_image.convert("RGB"))
        masks = mask_generator.generate(image)
        print(len(masks))
        # print(masks[0].keys())
        image_with_masks = show_anns_on_image(read_image.convert("RGBA"), masks)
        image_with_masks.save(f"{mask_save_dir}/{filename}")
        # with open(f'{json_save_dir}/{purename}.json', 'w', encoding='utf-8') as f:
        #     new_mask = []
        #     for m in masks:
        #         m['segmentation'] = m['segmentation'].tolist()
        #         new_mask.append(m)
        #     json.dump(new_mask, f, ensure_ascii=False)

            

if __name__ == '__main__':
    everything_infer()