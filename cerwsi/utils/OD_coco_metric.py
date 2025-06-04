from collections import defaultdict
from mmengine.evaluator import BaseMetric
from prettytable import PrettyTable
import numpy as np
import os
from mmengine.logging import MMLogger
from mmdet.evaluation import CocoMetric
from .metrics import print_confusion_matrix,calculate_metrics


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
        self.coco_metric.dataset_meta = dict(classes=classes)

    def format_pred2coco(self, data_sample, pred_bbox):
        pred_cocoformat = dict()
        pred_cocoformat['img_id'] = data_sample['img_id']
        pred_cocoformat['bboxes'] = np.array([preditem['bbox'] for preditem in pred_bbox])
        pred_cocoformat['scores'] = np.array([preditem['score'] for preditem in pred_bbox])
        pred_cocoformat['labels'] = np.array([preditem['cls']-1 for preditem in pred_bbox])
        
        filename = os.path.basename(data_sample['img_path'])
        roi_id = '_'.join(filename.split('_')[:2])
        square_coords = data_sample['extra_info']['square_coords']
        pred_cocoformat['roi_info'] = {
            'roi_id': roi_id,
            'square_coords': square_coords,
        }

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
            pred_bbox = [i for i in pred_bbox if i['score'] > 0.3]
            coco_result = self.format_pred2coco(datasample.to_dict(),pred_bbox)            
            result = dict(
                img_id = datasample.img_id,
                img_gt = bs_img_gt[bidx].item(),
                img_pred = bs_img_pred[bidx].item(),
                pred_bbox = pred_bbox,
                prefix = datasample.extra_info['prefix'], # pfefix: 'neg', 'partial_pos', 'total_pos'
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
        # 将 Tile 的检测结果整理回 RoI 以计算 mAP 等指标结果
        # roi_coco_results = self.remapTile2RoI(coco_results)
        det_metrics = self.coco_metric.compute_metrics(coco_results)
        result_table_2 = PrettyTable()
        result_table_2.field_names = list(det_metrics.keys())
        result_table_2.add_row(list(det_metrics.values()))
        result_metrics.update(det_metrics)

        '''计算图片二分类以及目标检测的一致性'''
        consistency_metrics = compute_consistency_metrics(results)
        result_metrics.update(consistency_metrics)
        
        cmstr = print_confusion_matrix(cm, print_flag=False)
        str_metric = '\n' + str(result_table_1) + '\n' + str(result_table_2) + '\n' + cmstr
        logger = MMLogger.get_instance(self.logger_name)
        logger.info(str_metric)
        
        return result_metrics

    def remapTile2RoI(self, coco_results):
        tile_predlist = [item[1] for item in coco_results]
        roi_tilelist = defaultdict(list)
        for tileItem in tile_predlist:
            roi_id = tileItem['roi_info']['roi_id']
            roi_tilelist[roi_id].append(tileItem)
        for roi_id,tilelist in roi_tilelist.items():
            gt_cocoformat = dict()
        
