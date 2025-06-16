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
    resultdir = 'log/WINDOW_SIZE_1000/smartccs_518_fusiontrain'
    with open(f'{resultdir}/pred_result.json', 'r', encoding='utf-8') as f:
        pred_datas = json.load(f)
    cfg = Config.fromfile(f'{resultdir}/config.py')
    coco = COCO(cfg.val_annojson)

    category_count = defaultdict(int)
    pId_total_count = defaultdict(int)
    pId_error_count = defaultdict(int)
    area_distribution = {'small': 0, 'medium': 0, 'large': 0}
    for predinfo in tqdm(pred_datas, ncols=80):
        image_id = predinfo['img_id']
        img_info = coco.loadImgs(image_id)[0]
        img_path = f"{cfg.img_dir}/{img_info['file_name']}"
        patientId = '_'.join(img_info['file_name'].split('/')[1].split('_')[:-2])

        if predinfo['img_gt']==1:
            pId_total_count[patientId] += 1

        # if (predinfo['img_gt'] != predinfo['img_pred']) and predinfo['img_gt']==1:
        if (predinfo['img_gt'] != predinfo['img_pred']):
            ann_ids = coco.getAnnIds(imgIds=[image_id])
            anns = coco.loadAnns(ann_ids)
            
            # 统计病人 slide 中 分错的 tile 数量
            pId_error_count[patientId] += 1
            # for ann in anns:
            #     cat_id = ann['category_id']
            #     bbox = ann['bbox']  # format: [x, y, w, h]
            #     area = bbox[2] * bbox[3]

            #     # 统计类别
            #     category_count[cat_id] += 1

            #     # 统计面积段
            #     if area < 32**2:
            #         area_distribution['small'] += 1
            #     elif area < 96**2:
            #         area_distribution['medium'] += 1
            #     else:
            #         area_distribution['large'] += 1
            
            # if random.random() < 0.2:
            #     save_dir = f'{resultdir}/FN'
            #     draw_image(coco, img_path, anns, save_dir)
            # if patientId == 'ZY_ONLINE_1_101':
            #     save_dir = f'{resultdir}/FN_ZY_ONLINE_1_101'
            #     draw_image(coco, img_path, anns, save_dir)
    
    cat_id_to_name = {cat['id']: cat['name'] for cat in coco.loadCats(list(category_count.keys()))}
    # print("BBox 类别分布：")
    # for cat_id, count in category_count.items():
    #     print(f"  - {cat_id_to_name[cat_id]} ({cat_id}): {count} 个")

    # print("\nBBox 面积分布：")
    # for area_type, count in area_distribution.items():
    #     print(f"  - {area_type}: {count} 个")
    
    sorted_pId_count = dict(sorted(pId_error_count.items(), key=lambda x: x[1], reverse=True))
    print(sum(list(sorted_pId_count.values())[:20]))
    print("\nPatient Error Tile 分布：")
    for pid, count in sorted_pId_count.items():
        if count > 5:
            print(f"  - {pid}: {count}/{pId_total_count[pid]} tiles")

if __name__ == "__main__":
    main()

'''
BBox 类别分布：
  - HSIL (5): 1587 个
  - ASC-US (2): 968 个
  - LSIL (3): 397 个
  - ASC-H (4): 798 个
  - AGC (1): 22 个

BBox 面积分布：
  - small: 331 个
  - medium: 2732 个
  - large: 709 个

  
+--------+--------------+-----------------+-----------------+
|  AUC   | img_accuracy | img_sensitivity | img_specificity |
+--------+--------------+-----------------+-----------------+
| 0.7255 |    0.6995    |      0.5963     |      0.7451     |
+--------+--------------+-----------------+-----------------+

+--------------------------+
|     confusion matrix     |
+-----+------+------+------+
|     |  0   |  1   | sum  |
+-----+------+------+------+
|  0  | 4983 | 1705 | 6688 |
|  1  | 1191 | 1759 | 2950 |
| sum | 6174 | 3464 | 9638 |
+-----+------+------+------+

Patient Error Tile 分布：
  - ZY_ONLINE_1_101: 164/166 tiles
  - ZY_ONLINE_1_195: 63/75 tiles
  - ZY_ONLINE_1_17: 49/116 tiles
  - ZY_ONLINE_1_1479: 44/87 tiles
  - ZY_ONLINE_1_21: 43/59 tiles
  - ZY_ONLINE_1_8: 22/40 tiles
  - JFSW_2_1504: 20/27 tiles
  - ZY_ONLINE_1_45: 18/24 tiles
  - ZY_ONLINE_1_198: 17/48 tiles
  - JFSW_2_242: 16/16 tiles
  - ZY_ONLINE_1_11: 16/26 tiles
  - JFSW_2_139: 16/16 tiles
  - JFSW_2_326: 15/15 tiles
  - ZY_ONLINE_1_135: 15/108 tiles
  - JFSW_2_271: 14/16 tiles
  - JFSW_2_321: 14/14 tiles
  - JFSW_2_259: 14/14 tiles
  - JFSW_2_250: 13/16 tiles
  - ZY_ONLINE_1_104: 13/60 tiles
  - JFSW_2_265: 12/14 tiles
  - ZY_ONLINE_1_65: 12/93 tiles
  - JFSW_2_239: 12/16 tiles
  - JFSW_2_93: 12/14 tiles
  - JFSW_2_1605: 11/11 tiles
  - JFSW_2_55: 11/15 tiles
  - JFSW_2_132: 10/12 tiles
  - JFSW_2_349: 10/12 tiles
  - JFSW_2_562: 9/9 tiles
  - ZY_ONLINE_1_43: 9/45 tiles
  - JFSW_2_356: 9/10 tiles
  - ZY_ONLINE_1_14: 9/14 tiles
  - JFSW_2_666: 8/8 tiles
  - JFSW_2_310: 8/18 tiles
  - JFSW_2_173: 7/14 tiles
  - JFSW_2_15: 7/7 tiles
  - JFSW_2_71: 7/9 tiles
  - JFSW_2_1364: 7/8 tiles
  - JFSW_2_302: 7/9 tiles
  - JFSW_2_67: 7/9 tiles
  - JFSW_2_1539: 6/6 tiles
  - JFSW_2_1375: 6/9 tiles
  - JFSW_2_1556: 6/10 tiles
  - JFSW_2_826: 6/10 tiles
  - JFSW_2_1573: 6/7 tiles
  - JFSW_2_1510: 6/17 tiles
  - JFSW_2_1374: 6/12 tiles
  - JFSW_2_1419: 6/7 tiles
  - JFSW_2_358: 6/6 tiles
  - JFSW_2_1581: 6/15 tiles
  - JFSW_2_273: 6/16 tiles
  - JFSW_2_1465: 6/16 tiles
  - JFSW_2_98: 6/13 tiles
'''