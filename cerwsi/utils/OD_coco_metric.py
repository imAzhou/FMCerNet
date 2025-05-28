from collections import defaultdict
import json
from sklearn.metrics import accuracy_score, confusion_matrix
from mmengine.evaluator import BaseMetric
from prettytable import PrettyTable
import numpy as np
import re
import torch
from mmdet.structures.mask import encode_mask_results
from mmengine.structures import InstanceData
from mmengine.logging import MMLogger
from mmdet.evaluation import CocoMetric

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
        "cm": cm
    }


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

class ImgODCOCOMetric(BaseMetric):
    '''
    预测每张图片的阴阳性，及具体病变位置bbox的目标检测结果 (COCO评测指标)
    '''
    def __init__(self, logger_name, save_result_dir, val_evaluator, classes) -> None:
        super(ImgODCOCOMetric, self).__init__()
        self.logger_name = logger_name
        self.save_result_dir = save_result_dir
        self.coco_metric = CocoMetric(**val_evaluator)
        self.coco_metric.cat_ids = range(len(classes))

    def format_pred2coco(self, data_sample,pred_bbox):
        pred_cocoformat = dict()
        pred_cocoformat['img_id'] = data_sample['img_id']
        pred_cocoformat['bboxes'] = np.array([preditem['bbox'] for preditem in pred_bbox])
        pred_cocoformat['scores'] = np.array([preditem['score'] for preditem in pred_bbox])
        pred_cocoformat['labels'] = np.array([preditem['cls'] for preditem in pred_bbox])
        # # encode mask to RLE
        # if 'masks' in pred:
        #     pred_cocoformat['masks'] = encode_mask_results(
        #         pred['masks'].detach().cpu().numpy()) if isinstance(
        #             pred['masks'], torch.Tensor) else pred['masks']
        # # some detectors use different scores for bbox and mask
        # if 'mask_scores' in pred:
        #     pred_cocoformat['mask_scores'] = pred['mask_scores'].cpu().numpy()
        
        gt_cocoformat = dict()
        gt_cocoformat['width'] = data_sample['ori_shape'][1]
        gt_cocoformat['height'] = data_sample['ori_shape'][0]
        gt_cocoformat['img_id'] = data_sample['img_id']

        return gt_cocoformat,pred_cocoformat

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
            datasample = data_samples['data_samples'][bidx]
            pred_bbox = data_samples['pred_bbox'][bidx]
            coco_result = self.format_pred2coco(datasample.to_dict(),pred_bbox)            
            result = dict(
                img_gt = bs_img_gt[bidx].item(),
                img_pred = bs_img_pred[bidx].item(),
                pred_bbox = pred_bbox,
                prefix = datasample.prefix, # pfefix: 'neg', 'partial_pos', 'total_pos'
                coco_result = coco_result
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
        cm = img_result['cm']
        del img_result['cm']
        for k,v in img_result.items():
            result_metrics['img_'+k] = v
        
        result_table_1 = PrettyTable()
        result_table_1.field_names = result_metrics.keys()
        result_table_1.add_row(result_metrics.values())

        '''计算目标检测的结果'''
        coco_results = [rs['coco_result'] for rs in results]
        det_metrics = self.coco_metric.compute_metrics(coco_results)
        result_table_2 = PrettyTable()
        result_table_2.field_names = list(det_metrics.keys())
        result_table_2.add_row(list(det_metrics.values()))
        result_metrics.update(det_metrics)

        '''计算图片二分类以及目标检测的一致性'''
        consistency_metrics = compute_consistency_metrics(results)
        result_metrics.update(consistency_metrics)
        
        str_metric = '\n' + str(result_table_1) + '\n' + str(result_table_2)
        logger = MMLogger.get_instance(self.logger_name)
        logger.info(str_metric)
        
        return result_metrics
