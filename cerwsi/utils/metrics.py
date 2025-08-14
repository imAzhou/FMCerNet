import json
from sklearn.metrics import (
    roc_curve, auc,
    accuracy_score, recall_score, f1_score, confusion_matrix,classification_report)
from prettytable import PrettyTable
import numpy as np
import torch
import torchmetrics
from mmengine.evaluator import BaseMetric
from mmpretrain.evaluation import MultiLabelMetric,SingleLabelMetric
from mmengine.logging import MMLogger

def calculate_metrics(y_true, y_pred):
    # 准确率 (Accuracy)
    accuracy = accuracy_score(y_true, y_pred)

    # 特异性 (Specificity)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = (tn / (tn + fp)).item()
    sensitivity = (tp / (tp + fn)).item()

    return {
        "accuracy": round(accuracy, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        'cm': cm
    }

def print_confusion_matrix(cm, print_flag=True):
    result_table = PrettyTable(title='confusion matrix')
    row_sums = np.sum(cm, axis=1).reshape(-1, 1)
    col_sums = np.sum(cm, axis=0).reshape(1, -1)
    # 构建一个扩展的矩阵，包含混淆矩阵和行列求和
    cm_with_sums = np.vstack([np.hstack([cm, row_sums]), np.hstack([col_sums, [[np.sum(cm)]]])])

    result_table.field_names = ['','0','1','sum']
    result_table.add_row(['0'] + list(cm_with_sums[0]))
    result_table.add_row(['1'] + list(cm_with_sums[1]))
    result_table.add_row(['sum'] + list(cm_with_sums[2]))
    if print_flag:
        print(result_table)

    return str(result_table)

class BinaryMetric(BaseMetric):
    '''
    只需预测图片阴阳的概率
    '''
    def __init__(self, logger_name, thr=0.3, save_result_dir=None) -> None:
        self.thr = thr
        self.logger_name = logger_name
        self.save_result_dir = save_result_dir
        super(BinaryMetric, self).__init__()

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
        bs_img_pred = (data_samples['img_probs'] > self.thr).int()
        
        bs = bs_img_gt.shape[0]

        for bidx in range(bs):
            result = dict(
                img_id = data_samples['data_samples'][bidx].img_id,
                img_gt = bs_img_gt[bidx].item(),
                img_pred = bs_img_pred[bidx].item(),
                img_probs = data_samples['img_probs'][bidx].item()
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

        img_gt = [rs['img_gt'] for rs in results]
        img_pred = [rs['img_pred'] for rs in results]
        img_probs = [rs['img_probs'] for rs in results]
 
        fpr, tpr, thresholds = roc_curve(img_gt, img_probs)
        result_metrics['AUC'] = round((auc(fpr, tpr)).item(), 4)
        img_result = calculate_metrics(img_gt,img_pred)
        cm = img_result['cm']
        del img_result['cm']
        for k,v in img_result.items():
            result_metrics['img_'+k] = v
        
        result_table_1 = PrettyTable()
        result_table_1.field_names = result_metrics.keys()
        result_table_1.add_row(result_metrics.values())

        cmstr = print_confusion_matrix(cm, print_flag=False)
        str_metric = '\n' + str(result_table_1) + '\n'+ '\n' + cmstr
        
        logger = MMLogger.get_instance(self.logger_name)
        logger.info(str_metric)

        if self.save_result_dir is not None:
            with open(f'{self.save_result_dir}/pred_result.json', 'w', encoding='utf-8') as f:
                json.dump(results, f)

        return result_metrics

class ExtendMultiLabelMetric(MultiLabelMetric):
    '''
    同时预测图片阴阳概率和含每个阳性类别的概率
    '''
    def __init__(self, thr, num_classes, logger_name) -> None:
        super(ExtendMultiLabelMetric, self).__init__(thr=thr)
        self.num_classes = num_classes
        self.logger_name = logger_name
        self.thr = thr

    def process(self, data_batch, data_samples):
        """Process one batch of data samples.

        The processed results should be stored in ``self.results``, which will
        be used to computed the metrics when all batches have been processed.

        Args:
            data_batch: A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """

        thr = self.thr if self.thr else 0.3
 
        for item in data_samples:
            pred_multi_label = [clsidx for clsidx,cls_score in enumerate(item['pos_prob']) if cls_score > thr]
            result = dict(
                img_gt = int(len(item['gt_label'])>0),
                img_pred = int(item['img_prob'] > thr),
                img_probs = item['img_prob'].item(),
                gt_multi_label = item['gt_label'],
                pred_multi_label = pred_multi_label,
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
        logger = MMLogger.get_instance(self.logger_name)
        result_metrics = dict()

        img_gt = [rs['img_gt'] for rs in results]
        img_pred = [rs['img_pred'] for rs in results]
        img_probs = [rs['img_probs'] for rs in results]
        
        fpr, tpr, thresholds = roc_curve(img_gt, img_probs)
        result_metrics['img_AUC'] = round((auc(fpr, tpr)).item(), 4)
        img_result = calculate_metrics(img_gt,img_pred)
        cm = img_result['cm']
        del img_result['cm']
        for k,v in img_result.items():
            result_metrics['img_'+k] = v
        
        result_table_1 = PrettyTable()
        result_table_1.field_names = result_metrics.keys()
        result_table_1.add_row(result_metrics.values())
        cmstr = print_confusion_matrix(cm, print_flag=False)
        str_metric = '\n' + str(result_table_1) + '\n' + cmstr
        logger.info(str_metric)

        gt_multi_label = [rs['gt_multi_label'] for rs in results]
        pred_multi_label = [rs['pred_multi_label'] for rs in results]
        clswise_metric_res = self.calculate(
            pred_multi_label,
            gt_multi_label,
            pred_indices=True,
            target_indices=True,
            average=None,
            num_classes=self.num_classes)

        def pack_results(precision, recall, f1_score, support):
            single_metrics = {}
            if 'precision' in self.items:
                single_metrics['precision'] = precision
            if 'recall' in self.items:
                single_metrics['recall'] = recall
            if 'f1-score' in self.items:
                single_metrics['f1-score'] = f1_score
            if 'support' in self.items:
                single_metrics['support'] = support
            return single_metrics
        
        suffix = '_classwise' if self.thr == 0.5 else f'_thr-{self.thr:.2f}_classwise'
        for k, v in pack_results(*clswise_metric_res).items():
            value = [round(i, 4) for i in v.detach().cpu().tolist()]
            result_metrics[k + suffix] = value
        
        macro_metric_res = self.calculate(
            pred_multi_label,
            gt_multi_label,
            pred_indices=True,
            target_indices=True,
            average='macro',
            num_classes=self.num_classes)
        for k, v in pack_results(*macro_metric_res).items():
            result_metrics[k] = round(v.item(), 4)
        
        logger.info(result_metrics)

        return result_metrics

class SlideMetric(BaseMetric):
    '''
    计算多类别: AUC, Acc, Cohen’s Kappa 系数, Precision, Recall, F1, Binary Sensitivity and Specificity
    '''
    def __init__(self, num_classes, logger_name) -> None:
        super(SlideMetric, self).__init__()
        self.num_classes = num_classes
        self.logger_name = logger_name
        kwargs = {'task': 'multiclass', 'num_classes': num_classes}
        self.AUROC = torchmetrics.AUROC(**kwargs, average = 'macro')
        self.metrics = torchmetrics.MetricCollection([
            torchmetrics.Accuracy(**kwargs, average='micro'),
            torchmetrics.CohenKappa(**kwargs),
            torchmetrics.F1Score(**kwargs, average = 'macro'),
            torchmetrics.Recall(**kwargs, average = 'macro'),
            torchmetrics.Precision(**kwargs, average = 'macro'),
        ])
        
    def process(self, data_batch, data_samples):
        """Process one batch of data samples.

        The processed results should be stored in ``self.results``, which will
        be used to computed the metrics when all batches have been processed.

        Args:
            data_batch: A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for item in data_samples:
            result = dict(
                pred_label = item['pred_label'].item(),
                pred_prob = item['pred_prob'].tolist(),
                gt_label = item['slide_label']
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
        logger = MMLogger.get_instance(self.logger_name)

        gt_label = torch.as_tensor([rs['gt_label'] for rs in results])
        pred_label = torch.as_tensor([rs['pred_label'] for rs in results])
        pred_prob = torch.as_tensor([rs['pred_prob'] for rs in results])

        result_table_1 = PrettyTable()
        result_metrics_1 = dict()
        auroc_score = self.AUROC(pred_prob, gt_label)   # tensor value
        result_metrics_1['AUROC'] = round(auroc_score.item(), 4)
        metric_scores = self.metrics(pred_label, gt_label)   # dict tensor value
        for k,v in metric_scores.items():
            result_metrics_1[k] = round(v.item(), 4)
        result_table_1.field_names = result_metrics_1.keys()
        result_table_1.add_row(result_metrics_1.values())

        result_table_2 = PrettyTable()
        result_metrics_2 = dict()
        binary_gt = [int(rs['gt_label']!=0) for rs in results]
        binary_pred = [int(rs['pred_label']!=0) for rs in results]
        img_result = calculate_metrics(binary_gt,binary_pred)
        cm = img_result['cm']
        del img_result['cm']
        for k,v in img_result.items():
            result_metrics_2['img_'+k] = v
        result_table_2.field_names = result_metrics_2.keys()
        result_table_2.add_row(result_metrics_2.values())    

        cmstr = print_confusion_matrix(cm, print_flag=False)
        str_metric = '\n' + str(result_table_1) + '\n'+ str(result_table_2) + '\n' + cmstr
        
        logger.info(str_metric)
        result = {**result_metrics_1, **result_metrics_2}
        logger.info(result)
        return result
