import pandas as pd
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import json
from prettytable import PrettyTable
from cerwsi.utils import calculate_metrics,print_confusion_matrix
from sklearn.metrics import classification_report

CLSMAP = {
    'NILM':'NILM',
    'ASC-US':'ASC-US',
    'LSIL':'LSIL',
    'ASC-H':'ASC-H',
    'HSIL':'HSIL',
    'SCC':'HSIL',
    'AGC':'AGC',
    'AGC-N':'AGC',
    'AGC-NOS':'AGC',
}
CLSNAME = ['NILM', 'ASC-US', 'LSIL', 'ASC-H', 'HSIL', 'AGC']
CLS_COLORS = {
    'NILM': '#2ca02c',  # 绿色
    'ASC-US': '#ff9999',  # 浅红
    'LSIL': '#FF69B4',  # 中浅红
    'ASC-H': '#FF1493',  # 中红
    'HSIL': '#8B008B',  # 深红
    'AGC': '#1f77b4'   # 蓝色
}
pos_ratio_bins = [0.0, 0.005, 0.01, 0.05, 0.10, 0.15, 0.30, 0.5, 1]
pos_thr = 0.5

def get_patientIds_GT(patientIds, csv_file):
    df_data = pd.read_csv(csv_file)
    gt = []
    for pid in tqdm(patientIds, ncols=80):
        filtered = df_data.loc[df_data['patientId'] == pid]
        if filtered.empty:
            print(f"No matching {pid} found.")
        patient_row = filtered.iloc[0]
        gt.append(CLSNAME.index(CLSMAP[patient_row.kfb_clsname]))
    return gt

def draw_pos_ratio_in_clsname(patientIds, y_clsid_gt, predInfo, pos_thr, fig_savepath):
    pos_ratio = []
    for pid in tqdm(patientIds, ncols=80):
        pos_pred = [i[1] > pos_thr for i in predInfo[pid]]
        pos_nums = sum(pos_pred)
        total_nums = len(pos_pred)
        pos_ratio.append(round(pos_nums/(total_nums+1e-6), 2))

    clsnames = [CLSNAME[idx] for idx in y_clsid_gt]
    bin_labels = [f"{(pos_ratio_bins[i]*100):.1f}%~{(pos_ratio_bins[i+1]*100):.1f}%" for i in range(len(pos_ratio_bins)-1)]
    df = pd.DataFrame({'clsnames': clsnames, 'pos_ratio': pos_ratio})
    df['bins'] = pd.cut(df['pos_ratio'], pos_ratio_bins, labels=bin_labels, include_lowest=True)
    grouped = df.groupby(['bins', 'clsnames'], observed=False).size().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(8, 6))
    bar_width = 0.3  # 设置柱体宽度
    bottom = np.zeros(len(bin_labels))  # 记录每个区间当前堆叠的高度
    for cls in CLSNAME:  # 确保按照 CLSNAME 顺序绘制
        if cls in grouped.columns:
            ax.bar(bin_labels, grouped[cls], bottom=bottom, width=bar_width, label=cls, color=CLS_COLORS[cls])
            bottom += grouped[cls].values  # 更新堆叠位置

    ax.set_xlabel("pos patch ratio", fontsize=12)
    ax.set_ylabel("slide nums", fontsize=12)
    ax.set_title("ratio distribution of each class", fontsize=14)
    ax.legend(title="clsname", fontsize=10)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(fig_savepath)

    table = PrettyTable()
    table.field_names = [" "] + bin_labels + ["合计"]  # 设置表头
    for cls in CLSNAME:  # 确保按照 CLSNAME 顺序打印
        values = [grouped.loc[interval, cls] if cls in grouped.columns else 0 for interval in bin_labels]
        row = [cls] + values + [sum(values)]
        table.add_row(row)
    print(table)

    return str(table)

def draw_senspc_in_ratio(patientIds, y_true, predInfo, pos_thr, fig_savepath):
    '''
    生成不同阳性patch数量占比阈值下的敏感性和特异性
    （根据每个patch的置信度，大于 pos_thr 则该 patch 为阳性）
    '''
    sensitivity, specificity = [],[]
    pos_ratio = []
    for pid in tqdm(patientIds, ncols=80, desc='Collecting pos ratio'):
        pos_pred = [i[1] > pos_thr for i in predInfo[pid]]
        pos_nums = sum(pos_pred)
        total_nums = len(pos_pred)
        pos_ratio.append(pos_nums/(total_nums+1e-6))
    for r_bin in pos_ratio_bins:
        y_pred = (np.array(pos_ratio) > r_bin).astype(int)
        result = calculate_metrics(y_true, y_pred)
        sensitivity.append(result['sensitivity'])
        specificity.append(result['specificity'])
    
    bar_width = 0.2
    indices = np.arange(len(pos_ratio_bins))
    plt.figure(figsize=(8, 6))
    bars1 = plt.bar(indices - bar_width/2, sensitivity, width=bar_width, color='red', label='Sensitivity', alpha=0.7)
    bars2 = plt.bar(indices + bar_width/2, specificity, width=bar_width, color='green', label='Specificity', alpha=0.7)

    for bar in bars1:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.2f}', ha='center', va='bottom')
    for bar in bars2:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.2f}', ha='center', va='bottom')

    plt.xlabel("Pos Patch Ratio")
    plt.ylabel("Values")
    plt.title("Sensitivity/Specificity in Differ Ratio")
    plt.xticks(indices, [f'{(b*100):.1f}%' for b in pos_ratio_bins])  # 设置横轴刻度
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.savefig(fig_savepath)

def draw_senspc_in_thr(y_true, y_pred, fig_savepath):
    '''
    生成不同 slide 阳性置信度下的敏感性和特异性
    （根据每个 slide 预测的置信度，y_pred 值在0到1之间，长度与y_true一致）
    '''
    thr_bins = [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
    sensitivity, specificity = [],[]
    for t_bin in thr_bins:
        y_pred_label = (np.array(y_pred) > t_bin).astype(int)
        result = calculate_metrics(y_true, y_pred_label)
        sensitivity.append(result['sensitivity'])
        specificity.append(result['specificity'])
    
    bar_width = 0.2
    indices = np.arange(len(thr_bins))
    plt.figure(figsize=(8, 6))
    bars1 = plt.bar(indices - bar_width/2, sensitivity, width=bar_width, color='red', label='Sensitivity', alpha=0.7)
    bars2 = plt.bar(indices + bar_width/2, specificity, width=bar_width, color='green', label='Specificity', alpha=0.7)

    for bar in bars1:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.2f}', ha='center', va='bottom')
    for bar in bars2:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.2f}', ha='center', va='bottom')

    plt.xlabel("Slide Pos Thr")
    plt.ylabel("Values")
    plt.title("Sensitivity and Specificity vs Thr Bins")
    plt.xticks(indices, [str(b) for b in thr_bins])  # 设置横轴刻度
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.savefig(fig_savepath)

def save_slide_result(csv_file, predInfo, pos_thr, slide_pos_ratio, save_path, positive_ratio_thr = 0.05):
    slide_pred, slide_gt = [],[]
    all_kfb_info = pd.read_csv(csv_file)
    for row in tqdm(all_kfb_info.itertuples(), total=len(all_kfb_info), ncols=80):
        pos_pred = [i[1] > pos_thr for i in predInfo[row.patientId]]
        p_path_num = sum(pos_pred)
        n_patch_num = len(pos_pred) - p_path_num
        p_ratio = p_path_num / (p_path_num + n_patch_num + 1e-6)    # 防止除0
        pred_clsid = int(p_ratio > positive_ratio_thr)
        
        slide_gt.append(row.kfb_clsid)
        slide_pred.append(pred_clsid)

    metric_result = calculate_metrics(slide_gt, slide_pred)
    cm = metric_result['cm']
    del metric_result['cm']
    result_table = PrettyTable()
    result_table.field_names = metric_result.keys()
    result_table.add_row(metric_result.values())
    print(result_table)
    str_cm = print_confusion_matrix(cm)
    report = classification_report(slide_gt, slide_pred, target_names=["Neg", "Pos"])
    print(report)

    txt_lines = [
        f'positive_ratio_thr = {positive_ratio_thr}\n\n',
        str(result_table),
        '\n\n',
        str_cm,
        '\n\n',
        report,
        '\n\n',
        slide_pos_ratio
    ]

    with open(save_path, 'w') as f:
        f.writelines(txt_lines)

if __name__ == '__main__':
    root_dir = 'log/l_cerscanv3/wscer_partial/2025_04_10_16_07_44'
    # csv_file = 'data_resource/slide_anno/0319/val.csv'
    csv_file = '/c22073/zly/datasets/CervicalDatasets/LCerScanv3/annofiles/val.csv'

    with open(f'{root_dir}/slide_pred_result.json', 'r') as f:
        predInfo = json.load(f)

    patientIds = list(predInfo.keys())
    y_clsid_gt = get_patientIds_GT(patientIds, csv_file)
    y_true = (np.array(y_clsid_gt) > 0).astype(int)

    fig_savedir = f'{root_dir}/standardized'
    os.makedirs(fig_savedir, exist_ok=True)
    slide_pos_ratio = draw_pos_ratio_in_clsname(patientIds, y_clsid_gt, predInfo, pos_thr, f'{fig_savedir}/pos_ratio_in_clsname.png')
    draw_senspc_in_ratio(patientIds, y_true, predInfo, pos_thr, f'{fig_savedir}/senspc_in_ratio.png')
    # draw_senspc_in_thr(y_true, y_pred, f'{fig_savedir}/senspc_in_thr.png')

    save_slide_result(csv_file, predInfo, pos_thr, slide_pos_ratio, f'{root_dir}/slide_pred_result.txt')

