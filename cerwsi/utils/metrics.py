import json
from sklearn.metrics import (
    roc_curve, auc,
    accuracy_score, recall_score, f1_score, confusion_matrix,classification_report)
from prettytable import PrettyTable
import numpy as np
import torch
import torchmetrics
from mmengine.evaluator import BaseMetric
from mmpretrain.evaluation import MultiLabelMetric,SingleLabelMetric,ConfusionMatrix
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
        thr = self.thr if self.thr else 0.3
 
        for item in data_samples:
            result = dict(
                img_gt = int(len(item['gt_label'])>0),
                img_pred = int(item['img_prob'] > thr),
                img_probs = item['img_prob'].item(),
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
        result_metrics['AUC'] = round((auc(fpr, tpr)).item(), 4)
        cohenkappa = torchmetrics.CohenKappa(task="multiclass", num_classes=2)
        ck_score = cohenkappa(torch.tensor(img_pred), torch.tensor(img_gt))
        result_metrics['CohenKappa'] = round(ck_score.item(), 4)

        img_result = calculate_metrics(img_gt,img_pred)
        cm = img_result['cm']
        del img_result['cm']
        for k,v in img_result.items():
            result_metrics[k] = v
        
        result_table_1 = PrettyTable()
        result_table_1.field_names = result_metrics.keys()
        result_table_1.add_row(result_metrics.values())

        cmstr = print_confusion_matrix(cm, print_flag=False)
        str_metric = '\n' + str(result_table_1) + '\n' + cmstr
        logger.info(str_metric)

        return result_metrics

class ExtendMultiLabelMetric(MultiLabelMetric):
    '''
    同时预测图片阴阳概率和含每个阳性类别的概率
    '''
    def __init__(self, thr, num_classes, logger_name, with_binary) -> None:
        super(ExtendMultiLabelMetric, self).__init__(thr=thr)
        self.num_classes = num_classes
        self.logger_name = logger_name
        self.thr = thr
        self.with_binary = with_binary

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
            if self.with_binary:
                result = dict(
                    img_gt = int(len(item['gt_label'])>0),
                    img_pred = int(item['img_prob'] > thr),
                    img_probs = item['img_prob'].item(),
                    gt_multi_label = item['gt_label'],
                    pred_multi_label = pred_multi_label,
                )
            else:
                img_pred = int(len(pred_multi_label) > 0)
                img_probs = max(item['pos_prob']).item()
                result = dict(
                    img_gt = int(len(item['gt_label'])>0),
                    img_pred = img_pred,
                    img_probs = img_probs,
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
        cohenkappa = torchmetrics.CohenKappa(task="multiclass", num_classes=2)
        ck_score = cohenkappa(torch.tensor(img_pred), torch.tensor(img_gt))
        result_metrics['img_CohenKappa'] = round(ck_score.item(), 4)

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

class ExtendSingleLabelMetric(SingleLabelMetric):
    '''
    预测图片属于每个类别的概率，并计算二分类指标
    '''
    def __init__(self, num_classes, logger_name) -> None:
        super(ExtendSingleLabelMetric, self).__init__()
        self.num_classes = num_classes
        self.logger_name = logger_name

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
                img_pn_gt = int(item['gt_label']>0),
                img_pn_pred = int(item['pred_label']>0),
                img_pn_prob = max(item['pred_prob']).item(),
                img_mc_gt = item['gt_label'].cpu(),
                img_mc_pred = item['pred_label'].cpu(),
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

        img_gt = [rs['img_pn_gt'] for rs in results]
        img_pred = [rs['img_pn_pred'] for rs in results]
        img_probs = [rs['img_pn_prob'] for rs in results]
        
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
        
        y_pred = [rs['img_mc_pred'] for rs in results]
        y_true = [rs['img_mc_gt'] for rs in results]
        precision, recall, f1_score, support = self.calculate(
            y_pred,
            y_true,
            num_classes=self.num_classes)
        result_metrics['mc_precision'] = round(precision.item(), 4)
        result_metrics['mc_recall'] = round(recall.item(), 4)
        result_metrics['mc_f1_score'] = round(f1_score.item(), 4)
        
        clswise_metric_res = self.calculate(
            y_pred,
            y_true,
            average=None,
            num_classes=self.num_classes)
        for k, v in pack_results(*clswise_metric_res).items():
            value = [round(i, 4) for i in v.detach().cpu().tolist()]
            result_metrics[k+'_classwise'] = value
 
        logger.info(result_metrics)

        return result_metrics

class MultiClsMetric(SingleLabelMetric):
    '''
    预测图片属于每个类别的概率，并计算混淆矩阵
    '''
    def __init__(self, num_classes, classes, logger_name) -> None:
        super(MultiClsMetric, self).__init__()
        self.num_classes = num_classes
        self.classes = classes
        self.logger_name = logger_name

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
                img_mc_gt = item['gt_label'].cpu(),
                img_mc_pred = item['pred_label'].cpu(),
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
        
        y_pred = [rs['img_mc_pred'] for rs in results]
        y_true = [rs['img_mc_gt'] for rs in results]
        precision, recall, f1_score, support = self.calculate(
            y_pred,
            y_true,
            num_classes=self.num_classes)
        result_metrics['mc_precision'] = round(precision.item(), 4)
        result_metrics['mc_recall'] = round(recall.item(), 4)
        result_metrics['mc_f1_score'] = round(f1_score.item(), 4)
        
        clswise_metric_res = self.calculate(
            y_pred,
            y_true,
            average=None,
            num_classes=self.num_classes)
        cw_p, cw_r, cw_f1, cw_s = clswise_metric_res
        metric_table = PrettyTable()
        metric_table.field_names = ["Class Name", "Precision", "Recall", "F1-Score", "Support"]
        metric_table.align["Class Name"] = "l"
        for i in range(self.num_classes):
            metric_table.add_row([
                self.classes[i], 
                f"{cw_p[i]:.2f}", 
                f"{cw_r[i]:.2f}", 
                f"{cw_f1[i]:.2f}", 
                int(cw_s[i])
            ])
        logger.info("\nClass-wise Metrics:\n" + metric_table.get_string())
        
        # mc_cmatrix: 2D tensor matrix
        mc_cmatrix = ConfusionMatrix.calculate(y_pred, y_true, num_classes=self.num_classes)
        mc_cmatrix = mc_cmatrix.cpu().numpy()
        cm_table = PrettyTable()
        # 表头：True \ Pred | Class1 | Class2 | ... | Total
        cm_table.field_names = ["True \ Pred"] + self.classes + ["Total"]
        cm_table.align["True \ Pred"] = "l"
        
        for i in range(self.num_classes):
            row_data = mc_cmatrix[i].tolist()
            row_total = int(sum(row_data)) # 这一行真实类别的样本总数
            
            # 构建行：类别名 + 各列预测值 + 总计
            row = [self.classes[i]] + row_data + [row_total]
            cm_table.add_row(row)
            
        logger.info("\nConfusion Matrix:\n" + cm_table.get_string())
        
        for k, v in pack_results(*clswise_metric_res).items():
            value = [round(i, 4) for i in v.detach().cpu().tolist()]
            result_metrics[k+'_classwise'] = value
        
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
        self.metrics = torchmetrics.MetricCollection({
            # overall metrics
            "acc_micro": torchmetrics.Accuracy(**kwargs, average='micro'),
            "cohen_kappa": torchmetrics.CohenKappa(**kwargs),
            "f1_macro": torchmetrics.F1Score(**kwargs, average='macro'),
            "recall_macro": torchmetrics.Recall(**kwargs, average='macro'),
            "precision_macro": torchmetrics.Precision(**kwargs, average='macro'),

            # per-class metrics
            "f1_per_class": torchmetrics.F1Score(**kwargs, average=None),
            "recall_per_class": torchmetrics.Recall(**kwargs, average=None),
            "precision_per_class": torchmetrics.Precision(**kwargs, average=None),
        })
        
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

        result_metrics_1 = dict()
        auroc_score = self.AUROC(pred_prob, gt_label)   # tensor value
        result_metrics_1['AUROC'] = round(auroc_score.item(), 4)
        metric_scores = self.metrics(pred_label, gt_label)   # dict tensor value
        for k,v in metric_scores.items():
            if 'per_class' in k:
                result_metrics_1[k] = [round(item_v.item(), 4) for item_v in v]
            else:
                result_metrics_1[k] = round(v.item(), 4)

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
        str_metric = '\n'+ str(result_table_2) + '\n' + cmstr
        
        logger.info(str_metric)
        result = {**result_metrics_1, **result_metrics_2}
        logger.info(result)
        return result

class AttriMetric(BaseMetric):

    def __init__(self, logger_name, num_attributes=10) -> None:
        super(AttriMetric, self).__init__()
        self.num_attributes = num_attributes
        self.logger_name = logger_name
        
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
                pred_label = item['pred_label'].tolist(),   #  (num_attributes, )
                # pred_prob = item['pred_logit'].tolist(),    #  (num_attributes, num_classes)
                gt_label = item['attr_v']   # list: [4,0,2,...]
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

        gt_labels = torch.tensor([rs['gt_label'] for rs in results])
        pred_labels = torch.tensor([rs['pred_label'] for rs in results])

        # correct_mask[i][j] = True 表示第 i 个样本的第 j 个属性预测正确
        correct_mask = (pred_labels == gt_labels)

        # --- 指标 1: Instance-level Accuracy (Exact Match Ratio) ---
        # 要求一行里所有属性都对 (all=True)（只有当一张图片的所有属性全部预测正确时，才算预测正确）
        instance_correct = correct_mask.all(dim=1)  # [N]
        instance_acc = instance_correct.float().mean().item()

        # --- 指标 2: Mean Attribute Accuracy (全局平均准确率) ---
        # 所有正确预测数 / 总元素数 (平均每张图能对几个属性？)
        mean_acc = correct_mask.float().mean().item()

        # --- 指标 3: Per-Attribute Accuracy (单属性准确率) ---
        # 对列求平均 -> [10] (分别计算每个属性的准确率)
        per_attr_acc = correct_mask.float().mean(dim=0).tolist()

        # 表格 A: 全局概览
        table_summary = PrettyTable()
        table_summary.field_names = ["Metric", "Value"]
        table_summary.add_row(["Instance Acc (Exact Match)", f"{instance_acc:.2%}"])
        table_summary.add_row(["Mean Attribute Acc", f"{mean_acc:.2%}"])

        # 表格 B: 各属性详情
        table_details = PrettyTable()
        # 动态生成表头: Attr_0, Attr_1, ...
        header = [f"Attr_{i}" for i in range(self.num_attributes)]
        table_details.field_names = header
        
        # 格式化数值为百分比字符串
        row_values = [f"{val:.2%}" for val in per_attr_acc]
        table_details.add_row(row_values)

        # 4. 输出日志
        log_str = "\n" + "=="*10 + " Attribute Evaluation " + "=="*10 + "\n"
        log_str += str(table_summary) + "\n"
        log_str += "Per-Attribute Details:\n"
        log_str += str(table_details)
        
        logger.info(log_str)

        # 5. 返回字典供外层调用 (如 TensorBoard 记录)
        eval_results = {
            "attr/instance_acc": instance_acc,
            "attr/mean_acc": mean_acc,
        }
        # 将每个属性的准确率也放入字典
        for i, acc in enumerate(per_attr_acc):
            eval_results[f"attr/acc_attr_{i}"] = acc
        logger.info(eval_results)
        
        return eval_results


