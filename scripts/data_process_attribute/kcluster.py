import os
os.environ["OPENBLAS_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pickle
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix
from scipy.optimize import linear_sum_assignment
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE

# --- 3. 使用 PrettyTable 打印分类报告 ---
def print_report_table(y_true, y_pred, class_names):
    # 获取字典格式的报告
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, digits=4)
    
    table = PrettyTable()
    table.field_names = ["Class", "Precision", "Recall", "F1-Score", "Support"]
    table.align["Class"] = "l"  # 左对齐类名
    
    # 添加各个类别的行
    for label in class_names:
        metrics = report_dict[label]
        table.add_row([
            label, 
            f"{metrics['precision']:.4f}", 
            f"{metrics['recall']:.4f}", 
            f"{metrics['f1-score']:.4f}", 
            int(metrics['support'])
        ])
    # 添加汇总行
    precision = report_dict['macro avg']['precision']
    recall = report_dict['macro avg']['recall']
    F1Score = report_dict['macro avg']['f1-score']
    table.add_row(["Macro_AVG", f"{precision:.4f}", f"{recall:.4f}", f"{F1Score:.4f}", int(report_dict['macro avg']['support'])])
    
    print("\n### Classification Report ###")
    print(table)

# --- 4. 使用 PrettyTable 打印混淆矩阵 ---
def print_confusion_matrix_table(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    
    table = PrettyTable()
    # 第一列是 True Label 的名称，后面是各个预测类的名称
    table.field_names = ["True \ Pred"] + class_names
    
    for i, row in enumerate(cm):
        # 将整行数据转换为字符串列表，并在最前面加上真实类名
        table.add_row([class_names[i]] + list(row))
    
    print("\n### Confusion Matrix (Rows: True, Cols: Pred) ###")
    print(table)

def TSNE_plot(y_true, X, classes, save_path):
    """
    使用 t-SNE 将高维特征 X 降维到 2D 空间，并根据真实标签 y_true 着色保存。
    
    参数:
    - y_true: 真实类别的 ID 列表 (numpy array)
    - X: 特征矩阵，例如模型输出的概率分布 (numpy array)
    - classes: 类别名称列表 (如 ['NILM', 'GEC', ...])
    - save_path: 图片保存的完整路径 (如 'results/tsne.png')
    """
    print(f"开始计算 t-SNE 降维 (样本数: {len(X)})...")
    
    # 2. 配置 t-SNE
    # perplexity 建议设为样本数的平方根左右，或者 30-50 之间的通用值
    tsne = TSNE(
        n_components=2, 
        perplexity=min(30, len(X)-1), 
        n_iter=1000, 
        random_state=42, 
        init='pca', 
        learning_rate='auto'
    )
    X_embedded = tsne.fit_transform(X)

    # 3. 整理绘图数据
    # 将 y_true 的 ID 映射为字符串类名，方便图例显示
    y_names = [classes[i] for i in y_true]
    df = pd.DataFrame({
        'TSNE-1': X_embedded[:, 0],
        'TSNE-2': X_embedded[:, 1],
        'Category': y_names
    })

    # 4. 绘图
    plt.figure(figsize=(10, 8))
    sns.set_style("whitegrid") # 使用白格背景，适合学术打印
    
    # 使用自定义配色方案
    palette = sns.color_palette("husl", len(classes))
    
    scatter_plot = sns.scatterplot(
        data=df,
        x='TSNE-1',
        y='TSNE-2',
        hue='Category',
        hue_order=classes, # 强制按照 classes 列表的顺序显示图例
        palette=palette,
        alpha=0.7,
        s=50,
        edgecolor='w',
        linewidth=0.5
    )

    # 5. 美化细节
    plt.title('t-SNE Visualization of Feature Distribution', fontsize=14, pad=15)
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    
    # 将图例放在外面，防止遮挡散点
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Classes', fontsize=10)
    
    # 6. 保存并关闭
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"t-SNE 图像已保存至: {save_path}")


def main():
    
    with open(f"{log_dir}/pred_result.pkl", "rb") as f:
        pred_result = pickle.load(f)

    # 1. 数据收集与聚类 (假设 pred_result, classes 已定义)
    X = np.array([sample.pred_prob_flatten.detach().cpu().numpy() for sample in pred_result])
    y_true = np.array([sample.cls_id for sample in pred_result])

    # 执行聚类
    kmeans = KMeans(n_clusters=len(classes), random_state=42, n_init=10)
    y_pred_cluster = kmeans.fit_predict(X)

    # 2. 标签映射 (匈牙利算法)
    w = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for i in range(y_true.size):
        w[y_pred_cluster[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    mapping = {row_ind[i]: col_ind[i] for i in range(len(row_ind))}
    y_pred_mapped = np.array([mapping[c] for c in y_pred_cluster])

    return y_true, y_pred_mapped,X
    

if __name__ == "__main__":
    classes = ['NILM', 'GEC', 'AGC', 'ASC-US', 'LSIL', 'ASC-H', 'HSIL']
    log_dir = 'log/attri_cls/sigmoid_lora/attri_predict'
    # 执行打印
    y_true, y_pred_mapped,X = main()
    print_report_table(y_true, y_pred_mapped, classes)
    print_confusion_matrix_table(y_true, y_pred_mapped, classes)
    save_path = f'{log_dir}/kcluster_tsne_logit.png'
    TSNE_plot(y_true, X, classes, save_path)
