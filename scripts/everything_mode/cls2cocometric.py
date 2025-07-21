import json
import os
import pickle
from pycocotools.coco import COCO
from mmdet.evaluation import CocoMetric
from tqdm import tqdm
import torch
from torchvision.ops import nms

def main():
    result_pkl = '/c23030/zly/codes/mmpretrain/work_dirs/SmartCCS/20250715_110328/pred_result.pkl'
    bboxes_path = 'data_resource/HMCHH/proposals_file/fold1_val_bboxes.json'
    with open(bboxes_path, 'r', encoding='utf-8') as f:
        proposal_bboxes = json.load(f)
    
    gt_jsonfile = 'data_resource/HMCHH/annofiles_roi/fold1_val.json'
    coco_gt = COCO(gt_jsonfile)
    coco_metric = CocoMetric(
        ann_file=gt_jsonfile,
        metric='bbox',
        classwise=False,
        # iou_thrs=[0.5],
        proposal_nums=(100, 300, 1000)
    )
    coco_metric.dataset_meta = dict(classes=['abnormal'])

    with open(result_pkl, 'rb') as f:
        pred_result = pickle.load(f)
    crop_predinfo = {}
    for imgpred in tqdm(pred_result, ncols=90, desc='Remap crop to image'):
        patientId,filename = imgpred['img_path'].split('/')[-2:]
        crop_predinfo[f'{patientId}_{filename}'] = imgpred
    
    iou_thresh = 0.5
    for imgid, proposals in tqdm(proposal_bboxes.items(), ncols=90, desc='Calc Metric'):
        imginfo = coco_gt.loadImgs([int(imgid)])[0]
        purename = imginfo['file_name'].split('.')[0]
        pred_bboxes,pred_scores = [],[]
        for proposalinfo in proposals:
            keyname = f'{purename}_{proposalinfo["crop_filename"]}'
            proposal_predinfo = crop_predinfo[keyname]
            pos_score = proposal_predinfo['pred_score'][1]
            x,y,w,h = proposalinfo['bbox']
            pred_bboxes.append([x,y,x+w,y+h])
            pred_scores.append(pos_score)

        pred_bboxes = torch.tensor(pred_bboxes, dtype=torch.float32)
        pred_scores = torch.tensor(pred_scores, dtype=torch.float32)
        # keep_indices = nms(pred_bboxes, pred_scores, iou_thresh)
        # pred_bboxes = pred_bboxes[keep_indices]
        # pred_scores = pred_scores[keep_indices]

        pred_instances = dict(
            bboxes=pred_bboxes, # x1,y1,x2,y2
            scores=pred_scores,
            labels=torch.zeros(len(pred_bboxes), dtype=torch.int32),
        )
        
        coco_metric.process(
            {},
            [dict(pred_instances=pred_instances, 
                img_id=imginfo['id'], ori_shape=(imginfo['width'], imginfo['height']))])

    eval_results = coco_metric.evaluate(size=len(coco_gt.imgs))
    print(eval_results)
    
if __name__ == "__main__":
    main()