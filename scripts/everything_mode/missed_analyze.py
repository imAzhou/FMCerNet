import json
import pickle
from pycocotools.coco import COCO
import numpy as np
from tqdm import tqdm
import cv2
import os

def bbox_iou(box1, box2):
    """计算两个bbox的IoU，输入为 (x1,y1,x2,y2)"""
    # 交集区域
    inter_x1 = np.maximum(box1[0], box2[0])
    inter_y1 = np.maximum(box1[1], box2[1])
    inter_x2 = np.minimum(box1[2], box2[2])
    inter_y2 = np.minimum(box1[3], box2[3])

    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    # 各自面积
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = area1 + area2 - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / union_area

import json
import pickle
from tqdm import tqdm
import numpy as np
from pycocotools.coco import COCO

def bbox_iou(box1, box2):
    """计算两个bbox的IoU，输入为 (x1,y1,x2,y2)"""
    inter_x1 = np.maximum(box1[0], box2[0])
    inter_y1 = np.maximum(box1[1], box2[1])
    inter_x2 = np.minimum(box1[2], box2[2])
    inter_y2 = np.minimum(box1[3], box2[3])

    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / union_area


def analyze_unfound_gt(gt_coco_jsonfile, pkl_filepath, save_txt_path, iou_thr=0.3):
    coco = COCO(gt_coco_jsonfile)
    with open(gt_coco_jsonfile, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    with open(pkl_filepath, 'rb') as f:
        loaded_data = pickle.load(f)

    classes = [i['name'] for i in gt_data['categories']]

    # 未被找到的GT记录
    unfound_stats = {
        'small': {cls: 0 for cls in classes},
        'medium': {cls: 0 for cls in classes},
        'large': {cls: 0 for cls in classes},
    }

    # 全部GT类别统计
    total_gt_stats = {cls: 0 for cls in classes}

    for imgitem in tqdm(gt_data['images'], ncols=80, desc='Eval Metric'):
        filename = imgitem['file_name']
        ann_ids = coco.getAnnIds(imgIds=[imgitem['id']])
        anns = coco.loadAnns(ann_ids)

        proposal_bboxes = loaded_data.get(f"train/{filename}", {}).get('bboxes', None)
        if proposal_bboxes is None:
            continue
        proposal_bboxes = np.array(proposal_bboxes, dtype=np.float32)

        for ann in anns:
            gt_bbox = ann['bbox']  # [x, y, w, h]
            gt_box = np.array([
                gt_bbox[0],
                gt_bbox[1],
                gt_bbox[0] + gt_bbox[2],
                gt_bbox[1] + gt_bbox[3]
            ])
            gt_area = ann['area']
            cat_name = coco.loadCats([ann['category_id']])[0]['name']

            total_gt_stats[cat_name] += 1  # 统计总GT数量

            # IoU计算
            ious = [bbox_iou(gt_box, p_box) for p_box in proposal_bboxes]
            max_iou = max(ious) if ious else 0

            if max_iou < iou_thr:  # 没找到的GT
                if gt_area < 32**2:
                    scale = 'small'
                elif gt_area < 96**2:
                    scale = 'medium'
                else:
                    scale = 'large'
                unfound_stats[scale][cat_name] += 1

    # ===== 构造输出文本 =====
    lines = []
    lines.append(f"GT JSON file: {gt_coco_jsonfile}")
    lines.append(f"PKL file: {pkl_filepath}")
    lines.append("")

    # 所有GT类别分布
    lines.append("================ 所有 GT 的类别分布 ================")
    total_gt_sum = sum(total_gt_stats.values())
    for cls, cnt in total_gt_stats.items():
        percent = 100 * cnt / total_gt_sum if total_gt_sum > 0 else 0
        lines.append(f"{cls:15s}: {cnt:6d}  ({percent:5.2f}%)")
    lines.append("")

    # 未找到的GT分布
    lines.append("================ 未被找到的 GT 统计结果 ================")
    for scale in ['small', 'medium', 'large']:
        total = sum(unfound_stats[scale].values())
        if total == 0:
            continue
        lines.append(f"\n[{scale.upper()}] 未被找到的总数: {total}")
        for cls, cnt in unfound_stats[scale].items():
            if cnt > 0:
                lines.append(f"  {cls:15s}: {cnt}")

    # ===== 保存到txt =====
    with open(save_txt_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

    print(f"\n✅ 结果已保存到: {save_txt_path}")
    return unfound_stats, total_gt_stats

def analyze_gt(gt_coco_jsonfile, save_txt_path):
    coco = COCO(gt_coco_jsonfile)
    with open(gt_coco_jsonfile, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)

    classes = [i['name'] for i in gt_data['categories']]

    # 全部GT类别统计
    total_gt_stats = {cls: 0 for cls in classes}

    for imgitem in tqdm(gt_data['images'], ncols=80, desc='Eval Metric'):
        ann_ids = coco.getAnnIds(imgIds=[imgitem['id']])
        anns = coco.loadAnns(ann_ids)

        for ann in anns:
            cat_name = coco.loadCats([ann['category_id']])[0]['name']
            total_gt_stats[cat_name] += 1  # 统计总GT数量

    # ===== 构造输出文本 =====
    lines = []
    lines.append(f"GT JSON file: {gt_coco_jsonfile}")
    lines.append("")

    # 所有GT类别分布
    lines.append("================ 所有 GT 的类别分布 ================")
    total_gt_sum = sum(total_gt_stats.values())
    for cls, cnt in total_gt_stats.items():
        percent = 100 * cnt / total_gt_sum if total_gt_sum > 0 else 0
        lines.append(f"{cls:15s}: {cnt:6d}  ({percent:5.2f}%)")
    lines.append("")

    # ===== 保存到txt =====
    with open(save_txt_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

    print(f"\n✅ 结果已保存到: {save_txt_path}")


def compute_iou(box, boxes):
    """
    box: [x1, y1, x2, y2]
    boxes: (N, 4)
    return: IoU array, shape (N,)
    """
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0, xx2 - xx1)
    inter_h = np.maximum(0, yy2 - yy1)
    inter = inter_w * inter_h

    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    union = area_box + area_boxes - inter
    iou = inter / np.maximum(union, 1e-6)
    return iou

def visual_missed(gt_coco_jsonfile, pkl_filepath, visual_savedir, iou_thr):
    coco_gt = COCO(gt_coco_jsonfile)
    with open(gt_coco_jsonfile, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    with open(pkl_filepath, 'rb') as f:
        loaded_data = pickle.load(f)
    for imgitem in tqdm(gt_data['images'], ncols=80, desc='Eval Metric'):
        filename = imgitem['file_name']
        img_path = f'{img_dir}/{filename}'
        ann_ids = coco_gt.getAnnIds(imgIds=[imgitem['id']])
        anns = coco_gt.loadAnns(ann_ids)
        # np.array, shape is (proposal_num, 4), 4 is (x1,y1,x2,y2)
        proposal_bboxes = loaded_data.get(f"train/{filename}", {}).get('bboxes', None)
        missed_bboxes = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            ann_box = np.array([x, y, x + w, y + h])
            ious = compute_iou(ann_box, proposal_bboxes)
            if np.max(ious) < iou_thr:
                missed_bboxes.append(ann_box)

        if len(missed_bboxes) > 0:
            img = cv2.imread(img_path)
            # 左图：漏检 GT（红框）
            left_img = img.copy()
            for box in missed_bboxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(left_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(left_img, f"Missed GT: {len(missed_bboxes)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # 右图：所有 proposal（绿框）
            right_img = img.copy()
            for box in proposal_bboxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(right_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(right_img, f"Proposals: {len(proposal_bboxes)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 拼接
            combined = np.concatenate((left_img, right_img), axis=1)

            save_path = os.path.join(visual_savedir, filename)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, combined)

if __name__ == "__main__":
    root_dir = 'data_resource/BCCD'
    img_dir = f'{root_dir}/train'
    gt_coco_jsonfile = f'{root_dir}/annofiles/train_annotations.coco.json'
    pkl_filepath = f'{root_dir}/train_proposal_maxDet300.pkl'
    proposal_savepath = f"{root_dir}/proposal_analyze.txt"
    gt_savepath = f"{root_dir}/gt_analyze.txt"
    iou_thr = 0.3
    # analyze_gt(gt_coco_jsonfile, gt_savepath)
    # analyze_unfound_gt(gt_coco_jsonfile, pkl_filepath, proposal_savepath, iou_thr)
    visual_savedir = f'statistic_results/cellpose_infer/BCCD_missed'
    visual_missed(gt_coco_jsonfile, pkl_filepath, visual_savedir, iou_thr)
    
