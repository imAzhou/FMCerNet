import json
import pickle
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
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


def DAP_softprob(X_prob, cls_attr_dist, classes):
    """
    基于属性概率分布和类别-属性映射规则，利用最大似然估计预测最终类别。
    
    Args:
        X_prob (np.ndarray): 形状为 (N_samples, 25) 的扁平化概率矩阵。
                             每一行是一个样本所有属性预测概率的拼接。
        cls_attr_dist (dict): 类别到属性合法索引的映射表。
                              格式: {'NILM': [[0], [0], ...], 'ASC-US': [[0,1], ...]}
        classes (list): 类别名称列表，用于将索引映射回类别名或确定输出顺序。
        
    Returns:
        y_pred (np.ndarray): 形状为 (N_samples, ) 的整数数组，表示预测类别的索引。
    """
    
    # 2. 计算每个属性在 X_prob 中的切片索引
    # 结果类似: [0, 6, 9, 12, 14, 19, 21, 23, 25]
    slice_indices = np.cumsum([0] + attribute_classes)
    n_samples = X_prob.shape[0]
    n_classes = len(classes)
    
    # 初始化得分矩阵 (样本数 x 类别数)
    # scores[i, c] 表示第 i 个样本属于第 c 个类别的对数概率得分
    scores = np.zeros((n_samples, n_classes))
    epsilon = 1e-10 # 防止 log(0)
    
    # 3. 核心计算：按属性遍历（向量化计算所有样本）
    # 外层循环次数固定为 8 (属性数量)，内层计算全是矩阵操作，速度极快
    for attr_idx, (start, end) in enumerate(zip(slice_indices[:-1], slice_indices[1:])):
        
        # 取出当前属性对应的概率矩阵部分，形状 (N_samples, 当前属性维度)
        # 例如 attr_0: (N, 6)
        curr_attr_probs = X_prob[:, start:end]
        
        # 遍历每个目标类别，计算该属性对该类别的贡献
        for cls_idx, cls_name in enumerate(classes):
            # 获取该类别在该属性下的合法索引列表 (例如 ASC-US 在 attr_4 是 [0, 2, 4])
            valid_indices = cls_attr_dist[cls_name][attr_idx]
            
            # --- 关键步骤 ---
            # 1. 提取合法索引对应的概率列
            # 2. 按行求和 (sum(axis=1))，得到 P(Attr_i is Valid | Class_c)
            # 结果形状: (N_samples, )
            prob_sum = np.sum(curr_attr_probs[:, valid_indices], axis=1)
            
            # 3. 取对数并累加到总分中
            # log(P1 * P2 * ...) = log(P1) + log(P2) + ...
            scores[:, cls_idx] += np.log(prob_sum + epsilon)
            
    # 4. 取最大得分对应的类别索引
    y_pred = np.argmax(scores, axis=1)
    
    return y_pred

if __name__ == "__main__":
    classes = ['NILM', 'GEC', 'AGC', 'ASC-US', 'LSIL', 'ASC-H', 'HSIL']
    attribute_classes = [6,3,3,2,5,2,2,2]
    # cls_attr_dist: dict(clsname1=[[0, 1], ..., [0, 2, 4], [0], [1]], clsname2=[...],)
    with open('data_resource/cell_attri/configs/cls_attri_dist.json', 'r', encoding='utf-8') as f:
        cls_attr_dist = json.load(f)
    
    log_dir = 'log/attri_cls/sigmoid_lora/2025_12_25_13_10_47'
    # log_dir = 'log/attri_cls/softmax_lora/2025_12_25_15_35_42'
    with open(f"{log_dir}/pred_result.pkl", "rb") as f:
        pred_result = pickle.load(f)
    # X_prob: (len(pred_result), (sum(attribute_classes)))
    X_prob = np.array([sample.pred_prob_flatten.detach().cpu().numpy() for sample in pred_result])
    # y_true: (len(pred_result), )
    y_true = np.array([sample.gt_label.cpu().numpy() for sample in pred_result])
    # y_pred: (len(pred_result), )
    y_pred = DAP_softprob(X_prob, cls_attr_dist, classes)
    print_report_table(y_true, y_pred, classes)
    print_confusion_matrix_table(y_true, y_pred, classes)
