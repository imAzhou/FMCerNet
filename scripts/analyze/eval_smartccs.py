import json
from mmengine.config import Config
from tqdm import tqdm
import torch
import os
import cv2
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import numpy as np
from cerwsi.datasets import load_data
from cerwsi.utils import build_evaluator,ImgODCOCOMetric
import random

def draw_image(coco, img_path, anns, predbbox, save_dir):
    filename = os.path.basename(img_path)
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 加载类别名称
    cat_id_to_name = {cat['id']: cat['name'] for cat in coco.loadCats(coco.getCatIds())}

    # -------- 左图：绘制 GT --------
    image_gt = image.copy()
    for ann in anns:
        bbox = ann['bbox']  # [x, y, w, h]
        x, y, w, h = map(int, bbox)
        category_id = ann['category_id']
        label = cat_id_to_name.get(category_id, str(category_id))
        cv2.rectangle(image_gt, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
        cv2.putText(image_gt, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # -------- 右图：绘制预测框 --------
    image_pred = image.copy()
    for pred in predbbox:
        x1, y1, x2, y2 = map(int, pred['bbox'])  # (x1, y1, x2, y2)
        cls_id = pred['cls']
        label = cat_id_to_name.get(cls_id, str(cls_id))
        cv2.rectangle(image_pred, (x1, y1), (x2, y2), color=(255, 165, 0), thickness=2)
        cv2.putText(image_pred, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 0, 200), 2)

    # -------- 拼接与保存 --------
    concat_image = np.concatenate([image_gt, image_pred], axis=1)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    plt.imsave(save_path, concat_image)

def main():
    pred_root_dir = '/c22073/zly/codes/Smart-CCS/Det-cell/infer_result'
    cfg = Config.fromfile('configs/dataset/mmdet/l_cerscanv1_dataset.py')
    #  smartccs: 1.ASC-US, 2.LSIL, 3.ASC-H, 4.HSIL, 5.SCC, and 6.AGC,
    label_map = [0,2,3,4,5,5,1]
    valloader = load_data(cfg, ['val'])
    val_evaluator = dict(
        ann_file=cfg.val_annojson,
        # metric='bbox',
        metric='proposal',
        classwise=False,
        iou_thrs=[0.5],
        # metric_items = ['mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l', 'AR@1000'],
        format_only=False,)
    coco = COCO(cfg.val_annojson)
    evaluator = build_evaluator([ImgODCOCOMetric('eval_smartccs',None,val_evaluator,cfg.classes)])
    for idx, sampled_batch in enumerate(tqdm(valloader, ncols=80)):
        pred_bboxes,img_probs = [],[]
        for datasample in sampled_batch['data_samples']:
            save_jsonname = f'{pred_root_dir}/{datasample.img_id}.json'
            with open(save_jsonname, 'r', encoding='utf-8') as f:
                predinfo = json.load(f)
            predbboxes,predlabels,predscores = predinfo['bbox'],predinfo['label'],predinfo['score']
            image_boxes = []
            for bbox,label,score in zip(predbboxes,predlabels,predscores):
                if score > 0.3:
                    x1, y1, w, h = bbox
                    image_boxes.append({
                        'bbox': [x1, y1, x1+w, y1+h],
                        'score': score,
                        'cls': label_map[label]
                    })
            # if datasample.diagnose == 0:
            #     image_boxes = []
            pred_bboxes.append(image_boxes)
            prob = 1. if len(image_boxes) > 0 else 0.
            img_probs.append(prob)
            
            if random.random() < 0.02:
                save_dir = f'statistic_results/smartccs_cell_detector'
                ann_ids = coco.getAnnIds(imgIds=[datasample.img_id])
                anns = coco.loadAnns(ann_ids)
                os.makedirs(save_dir, exist_ok=True, mode=0o777)
                draw_image(coco, datasample.img_path, anns, image_boxes, save_dir)

        sampled_batch['pred_bbox'] = pred_bboxes
        sampled_batch['img_probs'] = torch.Tensor(img_probs)
        evaluator.process(data_samples=[sampled_batch], data_batch=None)
    metrics = evaluator.evaluate(len(valloader.dataset))
    print(metrics)

if __name__ == "__main__":
    main()

'''
proposal score_thr = 0.3
 Average Precision  (AP) @[ IoU=0.50:0.50 | area=   all | maxDets=100 ] = 0.080
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.080
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.50 | area= small | maxDets=1000 ] = 0.014
 Average Precision  (AP) @[ IoU=0.50:0.50 | area=medium | maxDets=1000 ] = 0.065
 Average Precision  (AP) @[ IoU=0.50:0.50 | area= large | maxDets=1000 ] = 0.157
 Average Recall     (AR) @[ IoU=0.50:0.50 | area=   all | maxDets=100 ] = 0.520
 Average Recall     (AR) @[ IoU=0.50:0.50 | area=   all | maxDets=300 ] = 0.520
 Average Recall     (AR) @[ IoU=0.50:0.50 | area=   all | maxDets=1000 ] = 0.520
 Average Recall     (AR) @[ IoU=0.50:0.50 | area= small | maxDets=1000 ] = 0.327
 Average Recall     (AR) @[ IoU=0.50:0.50 | area=medium | maxDets=1000 ] = 0.482
 Average Recall     (AR) @[ IoU=0.50:0.50 | area= large | maxDets=1000 ] = 0.672

proposal score_thr = 0.3 with datasample.diagnose == 1
 Average Precision  (AP) @[ IoU=0.50:0.50 | area=   all | maxDets=100 ] = 0.187
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.187
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.50 | area= small | maxDets=1000 ] = 0.031
 Average Precision  (AP) @[ IoU=0.50:0.50 | area=medium | maxDets=1000 ] = 0.155
 Average Precision  (AP) @[ IoU=0.50:0.50 | area= large | maxDets=1000 ] = 0.361
 Average Recall     (AR) @[ IoU=0.50:0.50 | area=   all | maxDets=100 ] = 0.520
 Average Recall     (AR) @[ IoU=0.50:0.50 | area=   all | maxDets=300 ] = 0.520
 Average Recall     (AR) @[ IoU=0.50:0.50 | area=   all | maxDets=1000 ] = 0.520
 Average Recall     (AR) @[ IoU=0.50:0.50 | area= small | maxDets=1000 ] = 0.327
 Average Recall     (AR) @[ IoU=0.50:0.50 | area=medium | maxDets=1000 ] = 0.482
 Average Recall     (AR) @[ IoU=0.50:0.50 | area= large | maxDets=1000 ] = 0.672

bbox
 Average Precision  (AP) @[ IoU=0.50:0.50 | area=   all | maxDets=100 ] = 0.024
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.024
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.50 | area= small | maxDets=1000 ] = 0.002
 Average Precision  (AP) @[ IoU=0.50:0.50 | area=medium | maxDets=1000 ] = 0.012
 Average Precision  (AP) @[ IoU=0.50:0.50 | area= large | maxDets=1000 ] = 0.048
 Average Recall     (AR) @[ IoU=0.50:0.50 | area=   all | maxDets=100 ] = 0.173
 Average Recall     (AR) @[ IoU=0.50:0.50 | area=   all | maxDets=300 ] = 0.173
 Average Recall     (AR) @[ IoU=0.50:0.50 | area=   all | maxDets=1000 ] = 0.173
 Average Recall     (AR) @[ IoU=0.50:0.50 | area= small | maxDets=1000 ] = 0.067
 Average Recall     (AR) @[ IoU=0.50:0.50 | area=medium | maxDets=1000 ] = 0.149
 Average Recall     (AR) @[ IoU=0.50:0.50 | area= large | maxDets=1000 ] = 0.241
'''