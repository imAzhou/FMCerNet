from collections import defaultdict
import json
from sklearn.metrics import accuracy_score, confusion_matrix
from mmengine.evaluator import BaseMetric
from prettytable import PrettyTable
import numpy as np
import re
from mmengine.logging import MMLogger
import torch
from mmdet.structures.bbox import bbox_mapping_back

def calculate_metrics(y_true, y_pred):
    # 准确率 (Accuracy)
    accuracy = accuracy_score(y_true, y_pred)

    # 特异性 (Specificity)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)

    return {
        "accuracy": round(accuracy, 4),
        "sensitivity": round(sensitivity.item(), 4),
        "specificity": round(specificity.item(), 4),
    }

def compute_iou(box1, box2):
    """计算两个框的 IOU"""
    x1, y1, x2, y2 = np.maximum(box1[0], box2[0]), np.maximum(box1[1], box2[1]), \
                     np.minimum(box1[2], box2[2]), np.minimum(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0

def match_bboxes(gt_bboxes, pred_bboxes, iou_thresh=0.3, use_cls=False):
    """匹配预测框与GT框，返回是否被命中的标记数组"""
    matched_gt = [False] * len(gt_bboxes)
    used_pred = [False] * len(pred_bboxes)

    for i, gt in enumerate(gt_bboxes):
        gt_box = gt['bbox']
        gt_cls = gt['cls'] if use_cls else None
        max_iou = 0
        matched = -1
        for j, pred in enumerate(pred_bboxes):
            if used_pred[j]:
                continue
            if use_cls and pred['cls'] != gt_cls:
                continue
            iou = compute_iou(gt_box, pred['bbox'])
            if iou > max_iou:
                max_iou = iou
                matched = j
        if max_iou >= iou_thresh and matched != -1:
            matched_gt[i] = True
            used_pred[matched] = True
    return matched_gt, used_pred

def compute_det_metrics(results):
    thresholds = [0.3, 0.5, 0.7]
    metric_out = {}

    for thresh in thresholds:
        # partial_pos
        total_tp_partial_nocls = 0
        total_gt_partial_nocls = 0
        total_tp_partial_cls = 0
        total_gt_partial_cls = 0

        # total_pos
        total_tp_total_nocls = 0
        total_fp_total_nocls = 0
        total_gt_total_nocls = 0

        total_tp_total_cls = 0
        total_fp_total_cls = 0
        total_gt_total_cls = 0

        for rs in results:
            prefix = rs['prefix']
            gt_bboxes = [
                {'bbox': bbox, 'cls': clsid}
                for bbox, clsid in zip(rs['gt_bbox']['bbox'], rs['gt_bbox']['clsids'])
            ]
            pred_bboxes = [b for b in rs['pred_bbox'] if b['score'] >= thresh]

            if prefix == 'partial_pos':
                # No class
                matched_nocls, _ = match_bboxes(gt_bboxes, pred_bboxes, iou_thresh=0.3, use_cls=False)
                total_tp_partial_nocls += sum(matched_nocls)
                total_gt_partial_nocls += len(gt_bboxes)

                # With class
                matched_cls, _ = match_bboxes(gt_bboxes, pred_bboxes, iou_thresh=0.3, use_cls=True)
                total_tp_partial_cls += sum(matched_cls)
                total_gt_partial_cls += len(gt_bboxes)

            elif prefix == 'total_pos':
                # No class
                matched_nocls, used_pred_nocls = match_bboxes(gt_bboxes, pred_bboxes, iou_thresh=0.3, use_cls=False)
                total_tp_total_nocls += sum(matched_nocls)
                total_fp_total_nocls += len(pred_bboxes) - sum(used_pred_nocls)
                total_gt_total_nocls += len(gt_bboxes)

                # With class
                matched_cls, used_pred_cls = match_bboxes(gt_bboxes, pred_bboxes, iou_thresh=0.3, use_cls=True)
                total_tp_total_cls += sum(matched_cls)
                total_fp_total_cls += len(pred_bboxes) - sum(used_pred_cls)
                total_gt_total_cls += len(gt_bboxes)

        # partial_pos recall
        recall_partial_nocls = total_tp_partial_nocls / total_gt_partial_nocls if total_gt_partial_nocls > 0 else 0.0
        recall_partial_cls = total_tp_partial_cls / total_gt_partial_cls if total_gt_partial_cls > 0 else 0.0
        metric_out[f'pp_binary_recall@{thresh}'] = round(recall_partial_nocls, 4)
        metric_out[f'pp_cls_recall@{thresh}'] = round(recall_partial_cls, 4)

        # total_pos no class
        recall_total_nocls = total_tp_total_nocls / total_gt_total_nocls if total_gt_total_nocls > 0 else 0.0
        precision_total_nocls = total_tp_total_nocls / (total_tp_total_nocls + total_fp_total_nocls) if (total_tp_total_nocls + total_fp_total_nocls) > 0 else 0.0
        metric_out[f'tp_binary_recall@{thresh}'] = round(recall_total_nocls, 4) 
        metric_out[f'tp_binary_precision@{thresh}'] = round(precision_total_nocls, 4)

        # total_pos with class
        recall_total_cls = total_tp_total_cls / total_gt_total_cls if total_gt_total_cls > 0 else 0.0
        precision_total_cls = total_tp_total_cls / (total_tp_total_cls + total_fp_total_cls) if (total_tp_total_cls + total_fp_total_cls) > 0 else 0.0
        metric_out[f'tp_cls_recall@{thresh}'] = round(recall_total_cls, 4)
        metric_out[f'tp_cls_precision@{thresh}'] = round(precision_total_cls, 4)

    return metric_out

def compute_consistency_metrics(results, thresholds=[0.3, 0.5, 0.7]):
    metric_out = {}
    for thresh in thresholds:
        total = 0
        consistent = 0

        for rs in results:
            img_pred = rs['img_pred']  # 0 或 1
            pred_bboxes = [b for b in rs['pred_bbox'] if b['score'] >= thresh]

            if img_pred == 1 and len(pred_bboxes) > 0:
                consistent += 1
            elif img_pred == 0 and len(pred_bboxes) == 0:
                consistent += 1
            total += 1

        metric_out[f'consistency@{thresh}'] = round(consistent / total, 4) if total > 0 else 0.0

    return metric_out

class ImgODMetric(BaseMetric):
    '''
    预测每张图片的阴阳性，及具体病变位置bbox的目标检测结果
    '''
    def __init__(self, logger_name, save_result_dir) -> None:
        super(ImgODMetric, self).__init__()
        self.logger_name = logger_name
        self.save_result_dir = save_result_dir

    def process(self, data_batch, data_samples):
        """Process one batch of data samples.

        The processed results should be stored in ``self.results``, which will
        be used to computed the metrics when all batches have been processed.

        Args:
            data_batch: A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        data_samples = data_samples[0]
        bs_img_gt = data_samples['image_labels']
        bs_img_pred = (data_samples['img_probs'] > 0.5).int()
        bs = bs_img_gt.shape[0]

        for bidx in range(bs):
            # metainfo = data_samples['metainfo'][bidx]
            datasample = data_samples['data_samples'][bidx]
            sf = datasample.scale_factor
            bbox_in_origin = bbox_mapping_back(
                datasample.gt_instances.bboxes.tensor,
                datasample.img_shape,
                (sf[0], sf[1], sf[0], sf[1]),
                flip=datasample.get('flip', False),
                flip_direction=datasample.get('flip_direction', None)
            )
            gt_bbox = {
                'bbox': bbox_in_origin.tolist(),  # [[x1, y1, x2, y2],...]
                # 'clsnames': metainfo['clsnames'],  # ['AGC',...]
                'clsids': datasample.gt_instances.labels.tolist(),  # [1,...]
            }
            result = dict(
                img_gt = bs_img_gt[bidx].item(),
                img_pred = bs_img_pred[bidx].item(),
                # pred_bbox: List[List[Dict]], 每张图对应一个预测框列表，每个框格式如下：
                # {'bbox': [x1, y1, x2, y2], 'score': float, 'cls': int}
                pred_bbox = data_samples['pred_bbox'][bidx],
                gt_bbox = gt_bbox,
                prefix = datasample.extra_info['prefix'] # pfefix: 'neg', 'partial_pos', 'total_pos'
            )

            # Save the result to `self.results`.
            self.results.append(result)

    def compute_metrics(self, results):
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        # NOTICE: don't access `self.results` from the method. `self.results`
        # are a list of results from multiple batch, while the input `results`
        # are the collected results.
        result_metrics = dict()

        # if self.save_result_dir is not None:
        #     with open(f'{self.save_result_dir}/pred_result.json', 'w', encoding='utf-8') as f:
        #         json.dump(results, f, ensure_ascii=False, indent=4)
        
        '''计算图片阴阳二分类的结果'''
        img_gt = [rs['img_gt'] for rs in results]
        img_pred = [rs['img_pred'] for rs in results]
        img_result = calculate_metrics(img_gt,img_pred)
        for k,v in img_result.items():
            result_metrics['img_'+k] = v
        
        result_table_1 = PrettyTable()
        result_table_1.field_names = result_metrics.keys()
        result_table_1.add_row(result_metrics.values())

        '''计算目标检测的结果'''
        det_metrics = compute_det_metrics(results)
        result_metrics.update(det_metrics)

        '''计算图片二分类以及目标检测的一致性'''
        consistency_metrics = compute_consistency_metrics(results)
        result_metrics.update(consistency_metrics)
        
        result_table_2 = PrettyTable()
        split_metrics = defaultdict(dict)
        str_metrics = {**det_metrics, **consistency_metrics}
        thresholds = sorted(set(re.search(r'@(.+)', k).group(1) for k in str_metrics if '@' in k))
        for k, v in str_metrics.items():
            match = re.match(r'(.+?)@(.+)', k)
            name, thresh = match.group(1), match.group(2)
            split_metrics[thresh][name] = v
        all_names = sorted({name for names in split_metrics.values() for name in names})
        field_names = []
        for t in thresholds:
            field_names.extend([f"Thr = {t} key", f"Thr = {t} value"])
        result_table_2.field_names = field_names
        
        # 填充每一行
        for name in all_names:
            row = []
            for t in thresholds:
                row.append(name)
                row.append(split_metrics[t].get(name, ""))
            result_table_2.add_row(row)
        
        str_metric = '\n' + str(result_table_1) + '\n' + str(result_table_2)
        logger = MMLogger.get_instance(self.logger_name)
        logger.info(str_metric)
        
        return result_metrics

if __name__ == '__main__':
    jsonpath = 'log/l_cerscanv4/wscer_partial/2025_04_21_00_22_04/pred_result.json'
    with open(jsonpath, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    det_metrics = compute_det_metrics(json_data)
    consistency_metrics = compute_consistency_metrics(json_data)
    print(det_metrics)
    print(consistency_metrics)
