from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import json
from pycocotools import mask as mask_utils
import numpy as np
from tqdm import tqdm
import torch
from PIL import Image
import os
import cv2
import multiprocessing
from multiprocessing import Pool
import glob
import random
from cerwsi.utils import set_seed
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def show_anns_on_image(image, anns, borders=True):
    if len(anns) == 0:
        return image
    
    img_np = np.array(image).astype(np.float32) / 255.0  # [H, W, 4]
    sorted_anns = sorted(anns, key=lambda x: x['area'], reverse=True)

    for ann in sorted_anns:
        m = ann['segmentation']
        m = mask_utils.decode(m)
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
    
def get_sam_autopredictor(device):
    
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
        points_per_batch=64,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.9,
        stability_score_offset=0.7,
        crop_n_layers=0,
        box_nms_thresh=0.8,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=25.0,
        use_m2m=False,
        output_mode='coco_rle',
        # multimask_output=False
    )
    
    return mask_generator

def everything_infer_test():
    device = torch.device('cuda:0')
    root_dir = 'data_resource/everything_mode'
    mask_save_dir = f'{root_dir}/image'
    json_save_dir = f'{root_dir}/resultjson'
    os.makedirs(mask_save_dir, exist_ok=True)
    os.makedirs(json_save_dir, exist_ok=True)

    mask_generator = get_sam_autopredictor(device)
    # img_root_dir = '/c22073/zly/datasets/CervicalDatasets/WINDOW_SIZE_1000/images'
    img_root_dir = 'data_resource/0511/WINDOW_SIZE_1000/images'
    demo_paths = [
        'partial_pos/JFSW_1_88_2403658000107_0.png',
        'partial_pos/JFSW_2_200_1218184888413_0.png',
        'total_pos/ZY_ONLINE_1_1479_3104345449406_762.png',
        'total_pos/WXL_1_20_2609020897867_6.png',
        'partial_pos/JFSW_2_2448_3158077713106_5.png',
        'neg/JFSW_2_429_1961931156489_7.png'
    ]
    
    for imgpath in tqdm(demo_paths, ncols=80):
        read_image = Image.open(f'{img_root_dir}/{imgpath}')
        filename = os.path.basename(imgpath)
        purename = filename.replace('.png', '')
        image = np.array(read_image.convert("RGB"))
        masks = mask_generator.generate(image)
        print(len(masks))
        image_with_masks = show_anns_on_image(read_image.convert("RGBA"), masks)
        image_with_masks.save(f"{mask_save_dir}/{filename}")
        with open(f'{json_save_dir}/{purename}.json', 'w', encoding='utf-8') as f:
            json.dump(masks, f, ensure_ascii=False)

def everything_infer(proc_id, device, json_save_dir, set_group):
    mask_generator = get_sam_autopredictor(device)

    for idx, img_path in enumerate(set_group):
        filename = img_path.split('/')[-1]
        purename = filename.replace('.png', '')
        prefix = img_path.split('/')[-2]
        save_jsonname = f'{json_save_dir}/{prefix}/{purename}.json'
        if os.path.exists(save_jsonname):
            continue
        read_image = Image.open(img_path)
        image = np.array(read_image.convert("RGB"))
        masks = mask_generator.generate(image)
        # print(len(masks))
        # image_with_masks = show_anns_on_image(read_image.convert("RGBA"), masks)
        # image_with_masks.save(f"{mask_save_dir}/{filename}")
        with open(save_jsonname, 'w', encoding='utf-8') as f:
            json.dump(masks, f, ensure_ascii=False)
        print(f'Core {proc_id} processed : {idx+1}/{len(set_group)}.')

def main():
    set_seed(1234)
    device = torch.device('cuda:0')
    data_root = 'data_resource/0511/WINDOW_SIZE_1000'
    json_save_dir = f'{data_root}/sam2Infer'
    for tag in ['neg', 'partial_pos', 'total_pos']:
        os.makedirs(f'{json_save_dir}/{tag}', exist_ok=True, mode=0o777)

    multiprocessing.set_start_method('spawn', force=True)
    
    all_imgpath = []
    with open(f'{data_root}/annofiles/fusiontrain_cocoformat.json', 'r', encoding='utf-8') as f:
        fusiontrain_data = json.load(f)
    all_imgpath.extend([f'{data_root}/images/{item["file_name"]}' for item in fusiontrain_data['images']])
    # with open(f'{data_root}/annofiles/val_cocoformat.json', 'r', encoding='utf-8') as f:
    #     val_data = json.load(f)
    # all_imgpath.extend([f'{data_root}/images/{item["file_name"]}' for item in val_data['images']])
    total_nums = len(all_imgpath)
    cpu_num = 4
    set_split = np.array_split(all_imgpath, cpu_num)
    print(f"Number of cores: {cpu_num}, total_nums: {total_nums}, set number of per core: {len(set_split[0])}")
    workers = Pool(processes=cpu_num)
    processes = []
    for proc_id, set_group in enumerate(set_split):
        p = workers.apply_async(everything_infer, (proc_id, device, json_save_dir, set_group))
        processes.append(p)

    for p in processes:
        p.get()
    workers.close()
    workers.join()

def test_infer():
    root_dir = 'data_resource/everything_mode'
    mask_save_dir = f'{root_dir}/test_infer_1'
    os.makedirs(mask_save_dir, exist_ok=True, mode=0o777)

    data_root = 'data_resource/0511/WINDOW_SIZE_1000'
    json_save_dir = f'{data_root}/sam2Infer'
    all_imgpath = []
    # with open(f'{data_root}/annofiles/fusiontrain_cocoformat.json', 'r', encoding='utf-8') as f:
    #     fusiontrain_data = json.load(f)
    # all_imgpath.extend([f'{data_root}/images/{item["file_name"]}' for item in fusiontrain_data['images']])
    with open(f'{data_root}/annofiles/val_cocoformat.json', 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    all_imgpath.extend([f'{data_root}/images/{item["file_name"]}' for item in val_data['images']])
    
    random.shuffle(all_imgpath)
    for idx, img_path in enumerate(all_imgpath[:100]):
        filename = img_path.split('/')[-1]
        purename = filename.replace('.png', '')
        prefix = img_path.split('/')[-2]
        save_jsonname = f'{json_save_dir}/{prefix}/{purename}.json'
        with open(save_jsonname, 'r', encoding='utf-8') as f:
            masks = json.load(f)
        
        read_image = Image.open(img_path)
        image_with_masks = show_anns_on_image(read_image.convert("RGBA"), masks)
        image_with_masks.save(f"{mask_save_dir}/{prefix}_{purename}.png")
        

if __name__ == '__main__':
    # everything_infer_test()
    # main()
    test_infer()