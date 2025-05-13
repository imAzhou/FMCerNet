import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from cerwsi.utils import KFBSlide,read_json_anno


def plot_detailed_boxplot(data1, data2, titles):
    """
    绘制两批数据的宽高箱线图，并在图中标记中位数、四分位数和异常值。

    Args:
        data1 (tuple): 第一批数据的 (宽列表, 高列表)。
        data2 (tuple): 第二批数据的 (宽列表, 高列表)。
    """
    datasets = [data1, data2]
    colors = ['lightblue', 'lightgreen']

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    for idx, (data, color, title) in enumerate(zip(datasets, colors, titles)):
        widths, heights = data

        # 计算均值
        mean_width, mean_height = np.mean(widths), np.mean(heights)

        # 绘制箱线图
        bp = axes[idx].boxplot([widths, heights], labels=['Widths', 'Heights'], notch=True, patch_artist=True,
                               boxprops=dict(facecolor=color, color=color),
                               medianprops=dict(color='red'),
                               whiskerprops=dict(color='blue'))

        # 标记中位数、四分位数和异常值
        for i, label in enumerate(['Widths', 'Heights']):
            # 获取四分位数、中位数和异常值
            q1, q3 = bp['boxes'][i].get_path().vertices[0:3, 1][1:]
            med = bp['medians'][i].get_ydata()[1]
            fliers = bp['fliers'][i].get_ydata()

            # 标注
            # axes[idx].text(i + 1, med, f'Median: {med:.2f}', ha='center', va='center', fontsize=9, color='red')
            # axes[idx].text(i + 1, q1, f'Q1: {q1:.2f}', ha='center', va='bottom', fontsize=9, color='blue')
            axes[idx].text(i + 1, q3, f'Q3: {q3:.2f}', ha='center', va='top', fontsize=9, color='blue')

            # 标记异常值
            for outlier in fliers:
                axes[idx].text(i + 1.1, outlier, f'{outlier:.2f}', fontsize=8, color='purple')

        # 单独设置每个子图的 Y 轴范围
        min_val = min(min(widths), min(heights), min(fliers, default=np.inf))
        max_val = max(max(widths), max(heights), max(fliers, default=-np.inf)) + 100
        axes[idx].set_ylim(min_val, max_val)

        # 设置标题
        axes[idx].set_title(f"{title}\nWidth Mean: {mean_width:.2f}, Height Mean: {mean_height:.2f}")
        axes[idx].set_ylabel('Value')

    plt.tight_layout()
    plt.savefig(fig_savepath)



if __name__ == '__main__':
    fig_savepath = f'statistic_results/0103/wh_distribution_all.png'
    data_root_dir = '/medical-data/data'
    POSITIVE_CLASS = ['ASC-US', 'LSIL', 'ASC-H', 'HSIL', 'SCC', 'AGC-NOS', 'AGC', 'AGC-N', 'AGC-FN']
    colors = plt.cm.tab10(np.linspace(0, 1, len(POSITIVE_CLASS)))[:, :3] * 255
    category_colors = {cat: tuple(map(int, color)) for cat, color in zip(POSITIVE_CLASS, colors)}

    df_jf1 = pd.read_csv('data_resource/cls_pn/group_csv/JFSW_1.csv')
    df_jf2 = pd.read_csv('data_resource/cls_pn/group_csv/JFSW_2.csv')
    df_jf = pd.concat([df_jf1, df_jf2], ignore_index=True)

    small_cell_wh, large_cell_wh = [],[]
    huge_bbox = 0

    for row in tqdm(df_jf.itertuples(index=True), total=len(df_jf), ncols=80):
        if not isinstance(row.json_path, str):
            continue
        json_path = f'{data_root_dir}/{row.json_path}'
        annos = read_json_anno(json_path)
        for ann in annos:
            sub_class = ann['sub_class']
            region = ann.get('region')
            w,h = abs(region['width']),abs(region['height'])
            if w <=20 or h<=20 or sub_class not in POSITIVE_CLASS:
                continue

            if w > 1000 or h > 1000:
                huge_bbox += 1
                continue
            
            if w<100 and h<100:
                small_cell_wh.append([w,h])
            else:
                large_cell_wh.append([w,h])


    single_data = (np.array(small_cell_wh)[:,0], np.array(small_cell_wh)[:,1])
    cluster_data = (np.array(large_cell_wh)[:,0], np.array(large_cell_wh)[:,1])

    plot_detailed_boxplot(single_data, cluster_data, ['small bbox', 'large bbox'])
    print(f'small bbox anno nums: {len(small_cell_wh)}, large bbox anno nums: {len(large_cell_wh)}, huge bbox nums: {huge_bbox}')

'''
small bbox anno nums: 110748, large bbox anno nums: 45226, huge bbox nums: 18
'''