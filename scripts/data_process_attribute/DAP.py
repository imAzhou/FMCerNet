import json
import pickle
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from scipy.spatial.distance import cdist
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

def DAP(X_prob, cls_attr_dist, classes, attribute_dims=[5,3,3,2,5,4,2,2]):
    """
    Distance-based Attribute Prediction (DAP) - 修正版
    
    Args:
        X_prob: (N, 26) 模型的预测概率向量
        cls_attr_dist: 字典，格式 {'NILM': [(k_val, attr_indices), ...], ...}
                       其中元组的第二个元素 attr_indices 才是长度为8的属性列表
        classes: 类别名称列表
        attribute_dims: 每个属性头的类别数量列表
        
    Returns:
        y_pred: (N,) 预测的类别索引
    """
    
    # 1. 准备 One-Hot 映射所需的 Offset
    # offsets = [0, 5, 8, 11, 13, 18, 22, 24]
    offsets = np.cumsum([0] + attribute_dims[:-1])
    total_dim = sum(attribute_dims) # 26
    
    # 2. 构建属性原型库
    prototypes = []      # 存放展开后的 One-Hot 向量 (M, 26)
    proto_labels = []    # 存放对应的类别索引 (M,)
    
    for class_idx, class_name in enumerate(classes):
        if class_name not in cls_attr_dist:
            continue
            
        combinations = cls_attr_dist[class_name]
        # attr_indices: 长度为8的属性索引列表 -> 用这个构建向量
        for attr_indices in combinations:
            # 初始化全0向量
            vec = np.zeros(total_dim)
            # 将长度为8的 indices 转为长度为26的 one-hot
            # attr_indices 类似 [0, 2, 1, 0, 4, 3, 1, 1]
            for i, attr_val in enumerate(attr_indices):
                # 加上偏移量，定位到在26维向量中的绝对位置
                idx = int(offsets[i] + attr_val)
                vec[idx] = 1.0
            
            prototypes.append(vec)
            proto_labels.append(class_idx)
            
    # 转为 numpy 数组
    prototypes = np.array(prototypes)     # (M, 26)
    proto_labels = np.array(proto_labels) # (M,)
    
    # 3. 计算相似度/距离
    # 方法 A: 欧氏距离 (距离越小越好)
    # dists = cdist(X_prob, prototypes, metric='euclidean')
    # best_match_indices = np.argmin(dists, axis=1)
    
    # 方法 B: 点积相似度 (如果你的 X_prob 已经经过 Softmax，点积通常效果也很好且更快)
    scores = np.dot(X_prob, prototypes.T) 
    best_match_indices = np.argmax(scores, axis=1)
    
    # 4. 映射回类别
    y_pred = proto_labels[best_match_indices]
    
    return y_pred

if __name__ == "__main__":
    classes = ['NILM', 'GEC', 'AGC', 'ASC-US', 'LSIL', 'ASC-H', 'HSIL']
    attribute_classes = [5,3,3,2,5,4,2,2]
    # cls_attr_dist: dict(clsname1=[(K,8)], clsname2=[...],)
    with open('data_resource/cell_attri/configs/cls_attrset.json', 'r', encoding='utf-8') as f:
        cls_attr_dist = json.load(f)
    
    log_dir = 'log/attri_cls/softmax_lora/2025_12_27_15_12_31'
    with open(f"{log_dir}/pred_result.pkl", "rb") as f:
        pred_result = pickle.load(f)
    # X_prob: (len(pred_result), (sum(attribute_classes)))
    X_prob = np.array([sample.pred_prob_flatten.detach().cpu().numpy() for sample in pred_result])
    # y_true: (len(pred_result), )
    y_true = np.array([sample.gt_label.cpu().numpy() for sample in pred_result])
    # y_pred: (len(pred_result), )
    y_pred = DAP(X_prob, cls_attr_dist, classes)
    print_report_table(y_true, y_pred, classes)
    print_confusion_matrix_table(y_true, y_pred, classes)

