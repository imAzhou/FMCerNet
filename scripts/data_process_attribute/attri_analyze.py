import json
import os
import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from collections import Counter
from tqdm import tqdm

# --- 1. 配置与数据加载 ---
def load_and_preprocess():
    classes = ['NILM', 'GEC', 'AGC', 'ASC-US', 'LSIL', 'ASC-H', 'HSIL']
    json_path = 'data_resource/cell_attri/cell_inst_named.json'

    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    attr_nums = 0

    all_samples = []  # 用于存储 (K, 11) 的数据
    print("Loading JSON data and building sample matrix...")
    for tile_list in tqdm(json_data.values(), ncols=80):
        for tileitem in tile_list:
            clsid = classes.index(tileitem['sub_class'])
            # 获取属性值
            attr_v = list(tileitem['attr_v'])  # 确保是深拷贝，避免修改原数据
            attr_nums = len(attr_v)
            attr_v.append(clsid)
            all_samples.append(attr_v)

    # 转换为 DataFrame
    column_names = [f'Attr_{i+1}' for i in range(attr_nums)] + ['SubClass']
    df = pd.DataFrame(all_samples, columns=column_names)
    return df

# --- 2. 统计函数定义 ---
def calculate_cramers_v(x, y):
    """计算 Cramer's V (对称)"""
    confusion_matrix = pd.crosstab(x, y)
    if confusion_matrix.empty: return 0
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    # 偏差修正
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    if min((kcorr-1), (rcorr-1)) <= 0: return 0
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

def calculate_theils_u(x, y):
    """计算 Theil's U: U(y|x) (非对称)"""
    total = len(y)
    if total == 0: return 0
    # 计算 H(y|x)
    xy_counts = Counter(list(zip(x, y)))
    x_counts = Counter(x)
    s_xy = 0
    for (val_x, val_y), count in xy_counts.items():
        p_xy = count / total
        p_x = x_counts[val_x] / total
        s_xy += p_xy * math.log(p_x / p_xy)
    # 计算 H(y)
    y_counts = Counter(y)
    s_y = -sum([(c/total) * math.log(c/total) for c in y_counts.values()])
    
    if s_y == 0: return 1
    return (s_y - s_xy) / s_y

# --- 3. 执行分析与绘图 ---

def main():
    # 数据加载
    df = load_and_preprocess()
    n_features = df.shape[1]
    cols = df.columns
    v_matrix = np.zeros((n_features, n_features))
    u_matrix = np.zeros((n_features, n_features))

    print("\nComputing correlation matrices (this may take a minute)...")
    for i in tqdm(range(n_features), desc="Outer loop"):
        for j in range(n_features):
            # i 是自变量 (Row), j 是因变量 (Col)
            v_matrix[i, j] = calculate_cramers_v(df.iloc[:, i], df.iloc[:, j])
            u_matrix[i, j] = calculate_theils_u(df.iloc[:, i], df.iloc[:, j])

    v_df = pd.DataFrame(v_matrix, index=cols, columns=cols)
    u_df = pd.DataFrame(u_matrix, index=cols, columns=cols)

    # 结果导出与保存
    output_dir = 'data_resource/cell_attri/statistic_result'
    os.makedirs(output_dir, exist_ok=True)

    # 1. 绘制 Cramer's V
    plt.figure(figsize=(12, 10))
    sns.heatmap(v_df, annot=True, fmt='.2f', cmap='YlGnBu')
    plt.title("Cramer's V: Symmetric Association between Attributes")
    plt.savefig(os.path.join(output_dir, 'cramers_v_heatmap.png'), dpi=300)
    
    # 2. 绘制 Theil's U
    plt.figure(figsize=(12, 10))
    sns.heatmap(u_df, annot=True, fmt='.2f', cmap='magma')
    plt.title("Theil's U: Uncertainty Coefficient (Row predicts Column)")
    plt.savefig(os.path.join(output_dir, 'theils_u_heatmap.png'), dpi=300)

    print(f"\nAnalysis complete! Results saved in '{output_dir}/' directory.")
    
    # 打印最重要的发现：哪些属性对 SubClass 预测能力最强
    print("\nRanking of Attributes by predictive power on SubClass (Theil's U):")
    rank = u_df['SubClass'].drop('SubClass').sort_values(ascending=False)
    print(rank)

if __name__ == "__main__":
    main()
    