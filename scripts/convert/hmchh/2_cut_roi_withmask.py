import os
from PIL import Image
import json
from scipy import sparse
from cerwsi.utils import random_cut_square,calc_relative_coord,is_bbox_inside,generate_cut_regions
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

CERTAIN_THR = 0.7
LEVEL = 0
WINDOW_SIZE = 512
STRIDE = 450

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
        plt.contour(mask, levels=[0.5], colors='lime', linewidths=2)
        
    plt.axis('off')
    plt.tight_layout()
    os.makedirs('statistic_results/HMCHH/patch_from_RoI', exist_ok=True, mode=0o777)
    plt.savefig(f'statistic_results/HMCHH/patch_from_RoI/{filename}')
    plt.close()

def gene_patch_jsonlist():
    with open('data_resource/HMCHH/annofiles/unify_ann.json', 'r', encoding='utf-8') as f:
        roi_data = json.load(f)

    npz_mask_save_dir = 'data_resource/HMCHH/roi_inst_mask'
    patchitems = []
    for idx, item in enumerate(tqdm(roi_data, ncols=80)):

        for RoIItem in item['annotations']:
            rx1,ry1,rx2,ry2 = RoIItem['region']
            rw,rh = int(rx2-rx1+0.5), int(ry2-ry1+0.5)
            purename = item['patientId'] + f'_{str(RoIItem["annid"])}'
            
            if len(RoIItem['children']) > 0:
                loader = np.load(f"{npz_mask_save_dir}/{purename}.npz")
                sparse_mask = sparse.coo_matrix((loader['data'], (loader['row'], loader['col'])), shape=loader['shape'])
                roi_mask = sparse_mask.toarray().astype(np.int16)
            else:
                roi_mask = np.zeros((rh,rw), dtype=np.int16)

            RoI_patch_idx = 0
            cut_points = generate_cut_regions((0,0), rw, rh, WINDOW_SIZE, STRIDE)

            # 以异常框为中心裁剪 1 patch 块
            for annitem in RoIItem['children']:
                annx1,anny1,annx2,anny2 = annitem['region']
                annwidth,annheight = annx2-annx1, anny2-anny1
                random_x1,random_y1 = random_cut_square([annx1,anny1,annwidth,annheight],WINDOW_SIZE)  # 在资源中的绝对坐标
                random_square_coords = [random_x1,random_y1,random_x1+WINDOW_SIZE,random_y1+WINDOW_SIZE]
                if is_bbox_inside(random_square_coords, RoIItem['region'], tolerance=0):    # 必须被RoI完全包裹
                    res_x1,res_y1,_,_ = calc_relative_coord(RoIItem['region'],random_square_coords)    # 在 RoI 中的相对坐标
                    cut_points.append((int(res_x1),int(res_y1)))
            # 在 RoI 中随机裁剪 2 patch 块
            for i in range(2):
                random_x1,random_y1 = random_cut_square([0,0,rw,rh],WINDOW_SIZE)  # 在 RoI 中的相对坐标
                cut_points.append((random_x1,random_y1))

            for iidx,rect_coords in enumerate(cut_points):
                x1,y1 = rect_coords
                x2,y2 = x1+WINDOW_SIZE,y1+WINDOW_SIZE
                patch_coords = [x1,y1,x2,y2]    # 在 RoI 中的相对坐标
                bboxes,clsnames,patch_mask = calc_patch_anns(patch_coords, RoIItem, roi_mask)
                if patch_mask.shape[0] != WINDOW_SIZE or patch_mask.shape[1] != WINDOW_SIZE:
                    print(f'patch_mask {patch_mask.shape} != {WINDOW_SIZE}')
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
                if len(bboxes) == 0:
                    pItem['prefix'] = 'neg'
                    pItem['diagnose'] = 0
                    pItem['maskfile'] = ''
                elif len(bboxes) > 0:
                    pItem['prefix'] = 'total_pos' if RoIItem['sub_class'] == 'RoI' else 'partial_pos'
                    pItem['diagnose'] = 1
                    pItem['maskfile'] = f'{purename}_{RoI_patch_idx}.npz'
                    np.savez_compressed(f'{patch_npz_save_dir}/{pItem["maskfile"]}', patch_mask=patch_mask)

                    # roi_img = Image.open(pItem['source_path'])
                    # patch_img = roi_img.crop(pItem['square_coords'])
                    # vis_patch_sample(patch_img, patch_mask, bboxes, pItem['filename'])

                patchitems.append(pItem)
                RoI_patch_idx += 1

    with open(f'{json_save_dir}/patches_in_RoI.json', 'w', encoding='utf-8') as f:
        json.dump(patchitems, f, ensure_ascii=False)

def calc_patch_anns(patch_coords, RoIItem, roi_mask):
    rpx1,rpy1,rpx2,rpy2 = patch_coords  # 相对 roi 的坐标
    patch_mask = roi_mask[rpy1:rpy2, rpx1:rpx2]
    new_patch_mask = np.zeros_like(patch_mask)
    ann_bboxes, ann_clsnames = [],[]
    annidx = np.unique(patch_mask)
    if np.sum(annidx) == 0:     # 是否只有背景（背景 annid 为 0）
        return ann_bboxes, ann_clsnames, new_patch_mask
    
    for aidx in annidx[1:]:
        annmask = patch_mask == aidx
        instmask = np.argwhere(annmask)
        by1,bx1 = instmask.min(axis=0)    # (y_min, x_min)
        by2,bx2 = instmask.max(axis=0)  # (y_max, x_max)
        bwidth,bheight = bx2-bx1, by2-by1
        if bwidth > 20 and bheight > 20:
            ann_bboxes.append(np.array([bx1,by1,bx2,by2]).tolist())
            annitem = RoIItem['children'][aidx-1]
            ann_clsnames.append(annitem['sub_class'])
            new_patch_mask[annmask] = len(ann_bboxes)   # id: 1,2,...
    
    if np.sum(new_patch_mask>0) < 50*50:    # 总的阳性病变面积太小则忽略
        return [], [], np.zeros_like(patch_mask)
    
    return ann_bboxes, ann_clsnames, new_patch_mask

def cut_patch_imgs():
    
    with open(f'{json_save_dir}/patches_in_RoI.json', 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    reload_patchlist = defaultdict(list)
    for patchinfo in tqdm(json_data, ncols=80):
        keyname = f"{patchinfo['patientId']}_{patchinfo['media_type']}"
        reload_patchlist[keyname].append(patchinfo)
    
    total_nums = len(reload_patchlist.keys())
    for (keyname, patchlist),idx in zip(reload_patchlist.items(), range(total_nums)):
        source_path = patchlist[0]['source_path']
        roi_img = Image.open(source_path)

        for patchinfo in tqdm(patchlist, ncols=80, desc=f'[{idx+1}/{total_nums}]Processing {keyname}'):
            # px1,py1,px2,py2 = patchinfo['square_coords']
            # w,h = px2-px1,py2-py1
            patch_img = roi_img.crop(patchinfo['square_coords'])
            patch_img.save(f'{img_save_dir}/{patchinfo["prefix"]}/{patchinfo["filename"]}')
            # patch_mask = np.load(f'{patch_npz_save_dir}/{patchinfo["maskfile"]}')['patch_mask']
            # vis_patch_sample(patch_img, patch_mask, patchinfo['bboxes'], patchinfo['filename'])



if __name__ == "__main__":
    patch_npz_save_dir = f'data_resource/HMCHH/WINDOW_SIZE_{WINDOW_SIZE}/patch_inst_mask'
    os.makedirs(patch_npz_save_dir, exist_ok=True, mode=0o777)
    json_save_dir = f'data_resource/HMCHH/WINDOW_SIZE_{WINDOW_SIZE}/ann_jsons'
    os.makedirs(json_save_dir, exist_ok=True, mode=0o777)

    gene_patch_jsonlist()

    img_save_dir = f'data_resource/HMCHH/WINDOW_SIZE_{WINDOW_SIZE}/images'
    # for tag in ['neg', 'partial_pos', 'total_pos']:
    #     os.makedirs(f'{img_save_dir}/{tag}', exist_ok=True, mode=0o777)
    # cut_patch_imgs()
