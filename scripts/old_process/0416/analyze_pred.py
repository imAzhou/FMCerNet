import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm

def analyze_bbox_distribution(data_list, output_path="bbox_analysis.png"):
    def calc_area_and_ratio(bbox):
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        area = w * h
        ratio = w / h if h != 0 else 0
        return area, ratio

    gt_areas, gt_ratios = [], []
    pred_areas, pred_ratios = [], []

    for item in data_list:
        for bbox in item["gt_bbox"]["bbox"]:
            area, ratio = calc_area_and_ratio(bbox)
            gt_areas.append(area)
            gt_ratios.append(ratio)
        for pred in item["pred_bbox"]:
            area, ratio = calc_area_and_ratio(pred["bbox"])
            pred_areas.append(area)
            pred_ratios.append(ratio)

    # ✅ 自定义 bins 与标签
    bins = [0, 100*100, 200*200, 400*400, float('inf')]
    bin_labels = ['0–100²', '100²–200²', '200²–400²', '>400²']
    x = np.arange(len(bin_labels))
    width = 0.6

    # ✅ 面积分布
    gt_hist, _ = np.histogram(gt_areas, bins=bins)
    pred_hist, _ = np.histogram(pred_areas, bins=bins)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # 1. GT 面积分布柱状图
    axs[0].bar(x, gt_hist, width, color='tab:blue')
    axs[0].set_title('GT Area Distribution')
    axs[0].set_xlabel('Area Range')
    axs[0].set_ylabel('Count')
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(bin_labels, rotation=45)

    # 2. Pred 面积分布柱状图
    axs[1].bar(x, pred_hist, width, color='tab:orange')
    axs[1].set_title('Pred Area Distribution')
    axs[1].set_xlabel('Area Range')
    axs[1].set_ylabel('Count')
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(bin_labels, rotation=45)

    # 3. 宽高比 violinplot
    sns.violinplot(
        data=[gt_ratios, pred_ratios],
        ax=axs[2],
        inner="quartile"
    )
    axs[2].set_title('Aspect Ratio Distribution (W/H)')
    axs[2].set_xticks([0, 1])
    axs[2].set_xticklabels(['GT', 'Pred'])
    max_ratio = max(gt_ratios + pred_ratios)
    axs[2].set_yticks(np.arange(0, int(max_ratio) + 2, 1))
    axs[2].set_ylabel('Aspect Ratio')

    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()

def main():
    jsonpath = 'log/l_cerscanv4/wscer_partial/2025_04_21_00_22_04/pred_result.json'
    with open(jsonpath, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    output_path="log/l_cerscanv4/wscer_partial/2025_04_21_00_22_04/bbox_distribution.png"
    analyze_bbox_distribution(json_data, output_path)

if __name__ == "__main__":
    main()