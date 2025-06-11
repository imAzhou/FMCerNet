import json
import os
import cv2
from mmengine.config import Config
from tqdm import tqdm
from pycocotools.coco import COCO
from collections import defaultdict
import matplotlib.pyplot as plt
import random

def draw_image(coco, img_path, anns, save_dir):
    filename = os.path.basename(img_path)
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 加载类别名称
    cat_id_to_name = {cat['id']: cat['name'] for cat in coco.loadCats(coco.getCatIds())}
    # 绘制 bbox 和类别
    for ann in anns:
        bbox = ann['bbox']  # [x, y, w, h]
        x, y, w, h = map(int, bbox)
        category_id = ann['category_id']
        label = cat_id_to_name[category_id]
        # 画矩形框
        cv2.rectangle(image, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
        # 写类别标签
        cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # 保存图像
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/{filename}"
    plt.imsave(save_path, image)

def main():
    resultdir = 'log/WINDOW_SIZE_1000/CHIEF/smartccs_518_fusiontrain'
    with open(f'{resultdir}/pred_result.json', 'r', encoding='utf-8') as f:
        pred_datas = json.load(f)
    cfg = Config.fromfile(f'{resultdir}/config.py')
    coco = COCO(cfg.val_annojson)

    category_count = defaultdict(int)
    area_distribution = {'small': 0, 'medium': 0, 'large': 0}
    for predinfo in tqdm(pred_datas, ncols=80):
        if (predinfo['img_gt'] != predinfo['img_pred']) and predinfo['img_gt']==1:
            image_id = predinfo['img_id']
            ann_ids = coco.getAnnIds(imgIds=[image_id])
            anns = coco.loadAnns(ann_ids)
            img_info = coco.loadImgs(image_id)[0]
            img_path = f"{cfg.img_dir}/{img_info['file_name']}"

            for ann in anns:
                cat_id = ann['category_id']
                bbox = ann['bbox']  # format: [x, y, w, h]
                area = bbox[2] * bbox[3]

                # 统计类别
                category_count[cat_id] += 1

                # 统计面积段
                if area < 32**2:
                    area_distribution['small'] += 1
                elif area < 96**2:
                    area_distribution['medium'] += 1
                else:
                    area_distribution['large'] += 1
            
            if random.random() < 0.1:
                save_dir = f'{resultdir}/FN'
                draw_image(coco, img_path, anns, save_dir)

    
    cat_id_to_name = {cat['id']: cat['name'] for cat in coco.loadCats(list(category_count.keys()))}
    print("BBox 类别分布：")
    for cat_id, count in category_count.items():
        print(f"  - {cat_id_to_name[cat_id]} ({cat_id}): {count} 个")

    print("\nBBox 面积分布：")
    for area_type, count in area_distribution.items():
        print(f"  - {area_type}: {count} 个")

if __name__ == "__main__":
    main()

'''
FN ann box 分析：
BBox 类别分布：
  - LSIL (3): 240 个
  - HSIL (5): 1333 个
  - ASC-US (2): 622 个
  - ASC-H (4): 529 个
  - AGC (1): 16 个

BBox 面积分布：
  - small: 270 个
  - medium: 2018 个
  - large: 452 个
'''