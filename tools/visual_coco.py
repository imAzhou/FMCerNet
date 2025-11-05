import os
import json
import cv2
from tqdm import tqdm
from pycocotools.coco import COCO

def visualize_coco_annotations(coco_json, img_prefix, save_dir='visual_results'):
    """
    可视化 COCO 格式标注的 bbox 和类别名称。

    Args:
        coco_json (str): COCO 格式的 json 文件路径。
        img_prefix (str): 图片所在文件夹路径。
        save_dir (str): 保存可视化结果的目录。
    """
    os.makedirs(save_dir, exist_ok=True)
    coco = COCO(coco_json)
    cats = coco.loadCats(coco.getCatIds())
    cat_id_to_name = {cat['id']: cat['name'] for cat in cats}

    for img_info in tqdm(coco.dataset['images'], ncols=80, desc='Visualizing'):
        img_path = os.path.join(img_prefix, img_info['file_name'])
        if not os.path.exists(img_path):
            print(f"[Warning] Image not found: {img_path}")
            continue
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"[Warning] Failed to load: {img_path}")
            continue

        ann_ids = coco.getAnnIds(imgIds=img_info['id'])
        anns = coco.loadAnns(ann_ids)

        for ann in anns:
            if 'bbox' not in ann:
                continue
            x, y, w, h = map(int, ann['bbox'])
            cat_name = cat_id_to_name.get(ann['category_id'], 'Unknown')
            color = (0, 255, 0)

            # 绘制 bbox
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

            # 绘制类别文字背景框
            label = f"{cat_name}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x, y - th - 2), (x + tw + 2, y), color, -1)
            cv2.putText(img, label, (x + 1, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        save_path = os.path.join(save_dir, img_info['file_name'])
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, img)

    print(f"✅ 可视化完成！结果已保存到：{save_dir}")


        
    
if __name__ == "__main__":
    coco_json = 'data_resource/BCCD/annofiles/train_annotations.coco.json'
    img_prefix = 'data_resource/BCCD/train'
    save_dir = 'statistic_results/visual_results/gt_BCCD'
    visualize_coco_annotations(coco_json, img_prefix, save_dir)
