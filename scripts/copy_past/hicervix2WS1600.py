import json
import os
import random
from tqdm import tqdm
import cv2
import numpy as np

targetjson = 'data_resource/WINDOW_SIZE_1600/hardsample_annofiles/multilable_hs_round1.json'
target_imgdir = 'data_resource/WINDOW_SIZE_1600/images'
pos_srcjson = 'scripts/copy_past/hicervix.json'
random_cnt = [1,3]
paste_type = 'otsu'  # poisson,otsu
prefix = f'paste_hicervix_{paste_type}'
pastimg_savedir = f'data_resource/WINDOW_SIZE_1600/images/{prefix}'
pasted_savejson = f'data_resource/WINDOW_SIZE_1600/hardsample_annofiles/hs_round1_hicervix_{paste_type}.json'
os.makedirs(pastimg_savedir, exist_ok=True, mode=0o777)

def poisson_blend_images(src_list, background_path, mode=cv2.MIXED_CLONE):
    """
    将一组源图像按面积从大到小排序，依次随机粘贴到背景图像上，并进行 Poisson blending.
    
    Args:
        src_list (list): 源图像路径列表
        background_path (str): 背景图像路径
        mode (int): 融合模式，默认 cv2.MIXED_CLONE
                    可选: cv2.NORMAL_CLONE, cv2.MIXED_CLONE, cv2.MONOCHROME_TRANSFER
    
    Returns:
        blended (np.ndarray): 融合后的图像
    """
    background = cv2.imread(background_path)
    H, W = background.shape[:2]
    sources = []
    for path in src_list:
        img = cv2.imread(path)
        h, w = img.shape[:2]
        sources.append((img, w*h))
    # 按面积从大到小排序
    sources.sort(key=lambda x: x[1], reverse=True)
    blended = background.copy()
    for src, _ in sources:
        h, w = src.shape[:2]
        src_mask = 255 * np.ones(src.shape, src.dtype)
        # 随机选择位置
        center_x = random.randint(w//2, W - w//2)
        center_y = random.randint(h//2, H - h//2)
        center = (center_x, center_y)
        # Poisson blending
        blended = cv2.seamlessClone(src, blended, src_mask, center, mode)
    return blended

def otsu_blend_images(src_list, background_path):
    """
    将一组源图像按面积从大到小排序，依次随机粘贴到背景图像上.
    
    Args:
        src_list (list): 源图像路径列表
        background_path (str): 背景图像路径
        mode (int): 融合模式，默认 cv2.MIXED_CLONE
                    可选: cv2.NORMAL_CLONE, cv2.MIXED_CLONE, cv2.MONOCHROME_TRANSFER
    
    Returns:
        blended (np.ndarray): 融合后的图像
    """
    background = cv2.imread(background_path)
    H, W = background.shape[:2]
    
    sources = []
    for path in src_list:
        img = cv2.imread(path)
        h, w = img.shape[:2]
        sources.append((img, w*h))
    
    # 按面积从大到小排序
    sources.sort(key=lambda x: x[1], reverse=True)
    blended = background.copy()
    for src, _ in sources:
        h, w = src.shape[:2]
        lab = cv2.cvtColor(src, cv2.COLOR_BGR2Lab)
        L_channel = lab[:, :, 0]
        # Otsu 分割
        _, src_mask = cv2.threshold(L_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        src_mask = 255 - src_mask

        # 随机选择位置
        center_x = random.randint(w//2, W - w//2)
        center_y = random.randint(h//2, H - h//2)
        # 计算目标区域在 blended 中的坐标
        x1 = center_x - w//2
        y1 = center_y - h//2
        x2 = x1 + w
        y2 = y1 + h
        # 保证坐标不超出边界
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, W), min(y2, H)
        # 调整 src 和 mask 尺寸以匹配边界
        src_crop = src[0:(y2-y1), 0:(x2-x1)]
        mask_crop = src_mask[0:(y2-y1), 0:(x2-x1)]
        # 将 mask 转成布尔数组
        mask_bool = mask_crop.astype(bool)
        # 直接覆盖
        blended[y1:y2, x1:x2][mask_bool] = src_crop[mask_bool]

    return blended


def main():
    with open(targetjson, 'r', encoding='utf-8') as f:
        target_data = json.load(f)
    with open(pos_srcjson, 'r', encoding='utf-8') as f:
        possrc_data = json.load(f)
    classes = target_data['metainfo']['classes']
    pastelist = []
    for item in tqdm(target_data['data_list'], ncols=80):
        if len(item['gt_label']) == 0 and random.random() > 0.5:
            # 从 possrc_data 中随机选择 k 个样本
            k = random.randint(random_cnt[0], random_cnt[1])
            sampled = random.sample(possrc_data, k)
            src_list = [i['imgpath'] for i in sampled]
            target_path = f'{target_imgdir}/{item["img_path"]}'
            if paste_type == 'poisson':
                pastimg = poisson_blend_images(src_list, target_path)
            elif paste_type == 'otsu':
                pastimg = otsu_blend_images(src_list, target_path)
            purename = 'phi_' + item["img_path"].split('/')[1]
            cv2.imwrite(f'{pastimg_savedir}/{purename}', pastimg)
            clslebels = [classes.index(i['clsname']) for i in sampled]
            clslebels = list(set(clslebels))
            pastelist.append({
                'img_path': f'{prefix}/{purename}',
                'gt_label': clslebels
            })
    target_data['data_list'].extend(pastelist)
    with open(pasted_savejson, 'w', encoding='utf-8') as f:
        json.dump(target_data, f, ensure_ascii=False)
        

if __name__ == "__main__":
    main()