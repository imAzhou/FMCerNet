import pickle
import os
import torch
import numpy as np
from tqdm import tqdm
from prettytable import PrettyTable
from collections import defaultdict
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score

def save_and_print(text, file_handle):
    """辅助函数：同时打印到控制台和写入文件"""
    print(text)
    file_handle.write(text + "\n")

def main():
    attribute_names = ["Nsize","Nstains","Nchromatin","Nregular","cytoplasm","arrangement","polarity","cellType"]
    log_dir = 'log/attri_cls/softmax_lora/2025_12_27_15_12_31'
    
    pkl_path = f"{log_dir}/pred_result.pkl"
    if not os.path.exists(pkl_path):
        print(f"Error: File not found {pkl_path}")
        return

    with open(pkl_path, "rb") as f:
        pred_result = pickle.load(f)
    
    # --- 数据容器初始化 ---
    subclass_stats = defaultdict(lambda: {'acc': 0, 'err': 0})
    
    attr_gts = [[] for _ in range(len(attribute_names))]
    attr_preds = [[] for _ in range(len(attribute_names))]

    # --- 遍历数据 ---
    print("Processing prediction results...")
    for cellitem in tqdm(pred_result, ncols=80):
        # 数据转换 (Tensor -> Numpy)
        gt_attr_v = cellitem.attr_v.cpu().numpy() if isinstance(cellitem.attr_v, torch.Tensor) else np.array(cellitem.attr_v)
        pred_label = cellitem.pred_label.cpu().numpy() if isinstance(cellitem.pred_label, torch.Tensor) else np.array(cellitem.pred_label)
        sub_class = cellitem.sub_class

        # Task 1 & 2: 统计 Instance Level 的全对/错误情况
        is_perfect_match = np.array_equal(gt_attr_v, pred_label)
        
        if is_perfect_match:
            subclass_stats[sub_class]['acc'] += 1
        else:
            subclass_stats[sub_class]['err'] += 1
        
        # Task 3: 收集每个属性的数据
        for i in range(len(attribute_names)):
            attr_gts[i].append(int(gt_attr_v[i]))
            attr_preds[i].append(int(pred_label[i]))

    # --- 输出与保存 ---
    output_file_path = f"{log_dir}/evaluation_statistics.txt"
    with open(output_file_path, "w", encoding="utf-8") as f_out:
        
        header = f"Evaluation Report\nLog Dir: {log_dir}\nTotal Samples: {len(pred_result)}\n" + "="*50
        save_and_print(header, f_out)

        # === 1. 合并分布统计 (Merged Distribution with Percentage) ===
        table = PrettyTable()
        table.title = "1. Instance Level Accuracy Distribution per Subclass"
        # 新增 acc percent 列
        table.field_names = ["sub_class", "inst_acc", "inst_error", "total", "acc percent"]
        
        table.align["sub_class"] = "l"
        table.align["inst_acc cnt"] = "r"
        table.align["inst_error cnt"] = "r"
        table.align["total"] = "r"
        table.align["acc percent"] = "r"

        total_acc = 0
        total_err = 0

        # 按样本总数降序排序
        sorted_subclasses = sorted(subclass_stats.items(), key=lambda x: x[1]['acc'] + x[1]['err'], reverse=True)

        for sub, counts in sorted_subclasses:
            acc_cnt = counts['acc']
            err_cnt = counts['err']
            row_total = acc_cnt + err_cnt
            
            # 计算百分比
            acc_rate = (acc_cnt / row_total * 100) if row_total > 0 else 0.0
            
            table.add_row([sub, acc_cnt, err_cnt, row_total, f"{acc_rate:.2f}%"])
            
            total_acc += acc_cnt
            total_err += err_cnt

        # 计算总的准确率
        total_samples = total_acc + total_err
        total_rate = (total_acc / total_samples * 100) if total_samples > 0 else 0.0

        # 添加分割线和总计行
        table.add_row(["---", "---", "---", "---", "---"])
        table.add_row(["TOTAL", total_acc, total_err, total_samples, f"{total_rate:.2f}%"])
        
        save_and_print(str(table), f_out)
        save_and_print("\n", f_out)

        # === 2. 每个属性的混淆矩阵与详细指标 ===
        save_and_print("2. Confusion Matrix & Metrics per Attribute", f_out)
        
        for i, attr_name in enumerate(attribute_names):
            y_true = attr_gts[i]
            y_pred = attr_preds[i]
            
            labels = sorted(list(set(y_true) | set(y_pred)))
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            
            acc = accuracy_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

            # --- 构建混淆矩阵表格 ---
            cm_table = PrettyTable()
            cm_table.title = f"Attribute: {attr_name}"
            field_names = ["GT \ Pred"] + [f"Pred_{l}" for l in labels]
            cm_table.field_names = field_names
            
            for idx, label in enumerate(labels):
                row = [f"GT_{label}"] + list(cm[idx])
                cm_table.add_row(row)
            
            save_and_print(str(cm_table), f_out)
            
            metrics_str = (
                f"Metrics for {attr_name}:\n"
                f"  - Accuracy:     {acc:.4f}\n"
                f"  - Macro Recall: {recall:.4f}\n"
                f"  - Macro F1:     {f1:.4f}\n"
            )
            save_and_print(metrics_str, f_out)

    print(f"\nStatistics saved to: {output_file_path}")

if __name__ == "__main__":
    main()