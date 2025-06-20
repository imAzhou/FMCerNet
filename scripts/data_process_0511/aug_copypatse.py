import json
from collections import defaultdict
from pycocotools.coco import COCO
from tqdm import tqdm
import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import copy
from pycocotools import mask as maskUtils
from cerwsi.utils import set_seed


def visualize_lesion_transfer(src_img, src_ann, tgt_img, tgt_tile, updated_img, updated_tile):
    save_name = '_'.join(tgt_tile["file_name"].split('/'))
    savepath = f'{visual_savedir}/{save_name}'
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(src_img)
    axs[0].set_title("Source Image")
    axs[0].axis('off')
    axs[1].imshow(tgt_img)
    axs[1].set_title("Target Image")
    axs[1].axis('off')
    axs[2].imshow(updated_img)
    axs[2].set_title("After Lesion Transfer")
    axs[2].axis('off')
    
    for ax, img_anns in zip(axs, [[src_ann], tgt_tile['anns'], updated_tile['anns']]):
        for ann in img_anns:
            mask = maskUtils.decode(ann['segmentation']).astype(np.uint8)
            ax.contour(mask, colors='r', linewidths=1)
            x, y, w, h = ann['bbox']
            rect = plt.Rectangle((x, y), w, h, linewidth=1, edgecolor='lime', facecolor='none')
            ax.add_patch(rect)
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()

def copy_lesion_to_target(source_img, _target_img, _tgt_tile, ann, ann_id_counter, img_id_counter,paste_cnt):
    target_img = copy.deepcopy(_target_img)
    tgt_tile = copy.deepcopy(_tgt_tile)
    H, W = target_img.shape[:2]
    seg = ann['segmentation']
    mx1,my1,mw,mh = ann['bbox']
    mx2,my2 = mx1 + mw, my1 + mh
    lesion_mask = maskUtils.decode(seg).astype(np.bool)
    crop_mask = lesion_mask[my1:my2+1, mx1:mx2+1]

    annid_map = np.zeros_like(lesion_mask).astype(np.uint8)
    for annid,a in enumerate(tgt_tile['anns']):
        existing_mask = maskUtils.decode(a['segmentation']).astype(np.bool)
        annid_map[existing_mask] = annid+1
    
    best_tx, best_ty = None, None  # 最后一次尝试的位置
    for _ in range(20):
        tx = random.randint(0, W - mw - 1)
        ty = random.randint(0, H - mh - 1)
        best_tx, best_ty = tx, ty  # 始终记录最新尝试的位置
        shifted_mask = np.zeros_like(lesion_mask)
        shifted_mask[ty:ty+mh+1, tx:tx+mw+1] = crop_mask
        if not np.any(shifted_mask & (annid_map>0)):  # 找到不重叠的就立即使用
            break
    
    tgt_flag_mask = np.zeros_like(lesion_mask)
    tgt_flag_mask[best_ty:best_ty+mh+1, best_tx:best_tx+mw+1] = crop_mask
    target_img[tgt_flag_mask] = source_img[lesion_mask]

    bbox = [best_tx, best_ty, mw, mh]
    rle = maskUtils.encode(np.asfortranarray(tgt_flag_mask))
    rle['counts'] = rle['counts'].decode('utf-8')
    new_ann = {
        'id': ann_id_counter,
        'image_id': img_id_counter,
        'category_id': ann['category_id'],
        'segmentation': rle,
        'area': ann['area'],
        'bbox': bbox,
        'iscrowd': 0
    }
    ann_id_counter+=1
    
    if tgt_tile['diagnose'] == 0:
        tgt_tile['diagnose'] = 1
        tgt_tile['anns'] = [new_ann]
    else:
        annid_map[tgt_flag_mask] = len(tgt_tile['anns'])+1  # 新的 lesion 可能会覆盖掉之前的 lesion
        update_annlist = []
        for annid in np.unique(annid_map)[1:-1]:    # 重制旧 lesion 列表
            annmask = annid_map == annid
            ytrue, xtrue = np.where(annmask)
            x1, y1, x2, y2 = int(min(xtrue)), int(min(ytrue)), int(max(xtrue)), int(max(ytrue))
            w,h = x2-x1, y2-y1
            if w<20 and h<20:   # 太小的丢掉
                continue
            
            old_ann = tgt_tile['anns'][annid-1]
            rle = maskUtils.encode(np.asfortranarray(annmask))
            rle['counts'] = rle['counts'].decode('utf-8')
            old_ann['segmentation'] = rle
            old_ann['bbox'] = [x1, y1, w,h]
            old_ann['area'] = w*h
            old_ann['image_id'] = img_id_counter
            old_ann['id'] = ann_id_counter
            update_annlist.append(old_ann)
            ann_id_counter += 1
        tgt_tile['anns'] = [*update_annlist, new_ann]
    
    tgt_tile['id'] = img_id_counter
    pId = '_'.join(tgt_tile['file_name'].split('/')[1].split('_')[:-2])
    tgt_tile['file_name'] = f'paste_pos/{pId}_paste_{paste_cnt}.png'
    tgt_tile['extra_info']['prefix'] = 'paste_pos'

    return target_img, tgt_tile, ann_id_counter


def main(visual_img_num):
    src_jsonfile = 'puretrain_cocoformat'
    dest_jsonfile = 'puretrain_aug_cocoformat'
    with open(f'{dataroot}/annofiles/{src_jsonfile}.json', 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    coco = COCO(f'{dataroot}/annofiles/{src_jsonfile}.json')
    img_rootdir = f'{dataroot}/images'

    patientTileList = defaultdict(list)
    for _patchinfo in tqdm(json_data['images'], ncols=80):
        patchinfo = copy.deepcopy(_patchinfo)
        pId = '_'.join(patchinfo['file_name'].split('/')[-1].split('_')[:-2])
        if patchinfo['diagnose'] == 0:
            patchinfo['anns'] = []
        else:
            image_id = patchinfo['id']
            ann_ids = coco.getAnnIds(imgIds=[image_id])
            patchinfo['anns'] = coco.loadAnns(ann_ids)
        patientTileList[pId].append(patchinfo)
    
    updated_images = []
    updated_annotations = []
    ann_id_counter = max(coco.anns.keys()) + 1
    img_id_counter = max(coco.imgs.keys()) + 1
    filtered_patientTileList = {    # 过滤出至少包含一张阳性 tile 的病人列表
        pId: tiles for pId, tiles in patientTileList.items()
        if any(tile['diagnose'] == 1 for tile in tiles)
    }   # len = 787

    init_cnt = img_id_counter
    for pId, tiles in tqdm(filtered_patientTileList.items(), ncols=80):
        # if img_id_counter - init_cnt > 100:
        #     break
        positive_tiles = [tile for tile in tiles if tile['diagnose'] == 1]
        negative_tiles = [tile for tile in tiles if tile['diagnose'] == 0]

        pos_sample_num = max(1, len(positive_tiles) // 2)
        neg_sample_num = max(1, len(negative_tiles) // 2)
        target_tiles = random.sample(positive_tiles, pos_sample_num)    # 待粘贴的目标 tiles
        if negative_tiles:
            target_tiles += random.sample(negative_tiles, neg_sample_num)

        all_anns = []
        imgId2tile = {tile['id']: tile for tile in tiles}
        for tile in positive_tiles:
            all_anns.extend(tile['anns'])
        if not all_anns:
            continue
        src_lesions = random.sample(all_anns, max(1, len(all_anns) // 3))   # 挑出至少一个 源lesion
 
        paste_cnt = 0
        visual_cnt = visual_img_num
        for ann in src_lesions:
            image_id = ann['image_id']
            tile = imgId2tile[image_id]     # 当前 lesion 所在的 tile 信息
            src_img = cv2.imread(f'{img_rootdir}/{tile["file_name"]}')
            src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
            noself_target_tiles = [item for item in target_tiles if item['id']!=ann['image_id']]
            tgt_tile = random.choice(noself_target_tiles)

            tgt_img_path = f'{img_rootdir}/{tgt_tile["file_name"]}'
            tgt_img = cv2.imread(tgt_img_path)
            tgt_img = cv2.cvtColor(tgt_img, cv2.COLOR_BGR2RGB)

            updated_img, updated_tile, ann_id_counter = copy_lesion_to_target(
                src_img, tgt_img, tgt_tile, ann, ann_id_counter, img_id_counter, paste_cnt
            )
            paste_cnt += 1
            img_id_counter += 1

            # 可视化
            # if visual_cnt != 0:
            #     visualize_lesion_transfer(src_img, ann, tgt_img, tgt_tile, updated_img, updated_tile)
            #     visual_cnt -= 1
            # else:
            #     break

            # 保存更新图像
            save_path = f'{pastimg_savedir}/{os.path.basename(updated_tile["file_name"])}'
            cv2.imwrite(save_path, cv2.cvtColor(updated_img, cv2.COLOR_RGB2BGR))
            updated_annotations.extend(updated_tile['anns'])
            del updated_tile['anns']
            updated_images.append(updated_tile)
    print(len(updated_images))
    updated_json = {
        'images': [*json_data['images'], *updated_images],
        'annotations': [*json_data['annotations'], *updated_annotations],
        'categories': json_data['categories']
    }
    with open(f'{dataroot}/annofiles/{dest_jsonfile}.json', 'w', encoding='utf-8') as f:
        json.dump(updated_json, f, ensure_ascii=False)

if __name__ == "__main__":
    dataroot = 'data_resource/0511/WINDOW_SIZE_512'
    pastimg_savedir = f'{dataroot}/images/paste_pos'
    os.makedirs(pastimg_savedir, exist_ok=True, mode=0o777)
    set_seed(666)
    visual_savedir = 'statistic_results/aug_copypaste'
    os.makedirs(visual_savedir, exist_ok=True, mode=0o777)
    visual_img_num_eachpid = 5
    main(visual_img_num_eachpid)
