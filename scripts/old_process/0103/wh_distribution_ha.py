import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import random
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from cerwsi.utils import KFBSlide,read_json_anno

def draw_OD(read_image, save_path, square_coords, inside_items):
    draw = ImageDraw.Draw(read_image)
    sq_x1,sq_y1,sq_w,sq_h = square_coords

    for box_item in inside_items:
        category = box_item.get('sub_class')
        region = box_item.get('region')
        x,y = region['x'],region['y']
        w,h = region['width'],region['height']
        x1, y1, x2, y2 = x,y,x+w,y+h
        x_min = max(sq_x1, x1) - sq_x1
        y_min = max(sq_y1, y1) - sq_y1
        x_max = min(sq_x1+sq_w, x2) - sq_x1
        y_max = min(sq_y1+sq_h, y2) - sq_y1
        
        color = category_colors.get(category, (255, 255, 255))
        draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=3)
        draw.text((x_min + 2, y_min - 15), category, fill=color)
    
    # 使用 matplotlib 添加 legend
    fig, ax = plt.subplots(figsize=(sq_w//100+1, sq_h//100+1), dpi=100)
    ax.imshow(np.array(read_image))
    ax.axis('off')  # 不显示坐标轴
    # 创建 legend
    patches = [
        mpatches.Patch(color=np.array(color) / 255.0, label=category)  # Matplotlib 支持归一化颜色
        for category, color in category_colors.items()
    ]
    ax.legend(handles=patches, loc='upper right', bbox_to_anchor=(1.35, 1), frameon=False)
    fig.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)


def plot_detailed_boxplot(data1, data2, titles):
    """
    绘制两批数据的宽高箱线图，并在图中标记中位数、四分位数和异常值。

    Args:
        data1 (tuple): 第一批数据的 (宽列表, 高列表)。
        data2 (tuple): 第二批数据的 (宽列表, 高列表)。
    """
    datasets = [data1, data2]
    colors = ['lightblue', 'lightgreen']

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

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
            axes[idx].text(i + 1, med, f'Median: {med:.2f}', ha='center', va='bottom', fontsize=9, color='red')
            axes[idx].text(i + 1, q1, f'Q1: {q1:.2f}', ha='center', va='bottom', fontsize=9, color='blue')
            axes[idx].text(i + 1, q3, f'Q3: {q3:.2f}', ha='center', va='bottom', fontsize=9, color='blue')

            # 标记异常值
            for outlier in fliers:
                axes[idx].text(i + 1.1, outlier, f'{outlier:.2f}', fontsize=8, color='purple')

        # 单独设置每个子图的 Y 轴范围
        min_val = min(min(widths), min(heights), min(fliers, default=np.inf)) - 10
        max_val = max(max(widths), max(heights), max(fliers, default=-np.inf)) + 10
        axes[idx].set_ylim(min_val, max_val)

        # 设置标题
        axes[idx].set_title(f"{title}\nWidth Mean: {mean_width:.2f}, Height Mean: {mean_height:.2f}")
        axes[idx].set_ylabel('Value')

    plt.tight_layout()
    plt.savefig(f'statistic_results/0103/wh_distribution.png')

def convert_ha(ha):
    def flatten(data):
        result = []
        for item in data:
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, list):
                result.extend(flatten(item))  # 递归调用
        return result
    return list(set(flatten(ha)))  # 去重


if __name__ == '__main__':
    data_root_dir = '/medical-data/data'
    POSITIVE_CLASS = ['ASC-US', 'LSIL', 'ASC-H', 'HSIL', 'SCC', 'AGC-NOS', 'AGC', 'AGC-N', 'AGC-FN']
    colors = plt.cm.tab10(np.linspace(0, 1, len(POSITIVE_CLASS)))[:, :3] * 255
    category_colors = {cat: tuple(map(int, color)) for cat, color in zip(POSITIVE_CLASS, colors)}

    df_jf1 = pd.read_csv('data_resource/cls_pn/group_csv/JFSW_1.csv')
    df_jf2 = pd.read_csv('data_resource/cls_pn/group_csv/JFSW_2.csv')
    df_jf = pd.concat([df_jf1, df_jf2], ignore_index=True)

    single_cell_wh, cluster_cell_wh = [],[]
    total_ha = []
    sce,cce = [],[]
    for row in tqdm(df_jf.itertuples(index=True), total=len(df_jf), ncols=80):
        if not isinstance(row.json_path, str):
            continue
        json_path = f'{data_root_dir}/{row.json_path}'
        annos = read_json_anno(json_path)
        for ann in annos:
            sub_class = ann['sub_class']
            region = ann.get('region')
            w,h = abs(region['width']),abs(region['height'])
            
            if w>20 and h>20 and 'hierarchical_annotation' in ann and sub_class in POSITIVE_CLASS:
                ha = convert_ha(ann['hierarchical_annotation'])
                if '单个细胞' in ha:
                    single_cell_wh.append([w,h])
                    if (w > 100 and h > 100):
                        sce.append(sub_class)
                        if len(sce) < 10:
                            slide = KFBSlide(f'{data_root_dir}/{row.kfb_path}')
                            x1 = min(ann['points'][0]['x'],ann['points'][1]['x'])
                            y1 = min(ann['points'][0]['y'],ann['points'][1]['y'])
                            save_path = f'statistic_results/0103/single_cell_example_{len(sce)}.png'
                            new_w,new_h = 500,500
                            minx,miny = x1-(new_w-w), y1-(new_h-h)
                            maxx,maxy = x1, y1
                            newx,newy = random.randint(int(minx),int(maxx)), random.randint(int(miny),int(maxy))
                            square_coords = [newx,newy,new_w,new_h]
                            location, level, size = (newx,newy,new_w), 0, (new_w,new_h)
                            read_result = Image.fromarray(slide.read_region(location, level, size))
                            draw_OD(read_result, save_path, square_coords, [ann])            
                elif '成团细胞' in ha:
                    cluster_cell_wh.append([w,h])

                    if (w < 100 and h < 100):
                        cce.append(sub_class)
                        if len(cce) < 10:
                            slide = KFBSlide(f'{data_root_dir}/{row.kfb_path}')
                            x1 = min(ann['points'][0]['x'],ann['points'][1]['x'])
                            y1 = min(ann['points'][0]['y'],ann['points'][1]['y'])
                            save_path = f'statistic_results/0103/cluster_cell_example_{len(cce)}.png'
                            new_w,new_h = 500,500
                            minx,miny = x1-(new_w-w), y1-(new_h-h)
                            maxx,maxy = x1, y1
                            newx,newy = random.randint(int(minx),int(maxx)), random.randint(int(miny),int(maxy))
                            square_coords = [newx,newy,new_w,new_h]
                            location, level, size = (newx,newy,new_w), 0, (new_w,new_h)
                            read_result = Image.fromarray(slide.read_region(location, level, size))
                            draw_OD(read_result, save_path, square_coords, [ann])

                # total_ha.extend(ha)
    
    # unique_ha = list(set(total_ha))
    # with open(f'statistic_results/0103/unique_ha.txt', 'w') as f:
    #     unique_ha_lines = [f'{i}\n' for i in unique_ha]
    #     f.writelines(unique_ha_lines)
    
    single_data = (np.array(single_cell_wh)[:,0], np.array(single_cell_wh)[:,1])
    cluster_data = (np.array(cluster_cell_wh)[:,0], np.array(cluster_cell_wh)[:,1])

    plot_detailed_boxplot(single_data, cluster_data, ['single cell', 'cluster cell'])
    print(f'single cell anno nums: {len(single_cell_wh)}, cluster cells anno nums: {len(cluster_cell_wh)}')
    print(f'large single cell nums:{len(sce)} clsname:{list(set(sce))}')
    print(f'small cluster cell nums:{len(cce)} clsname: {list(set(cce))}')

'''
single cell anno nums: 62432, cluster cells anno nums: 35704
w > 100 and h > 100
large single cell nums:1870 clsname:['LSIL', 'HSIL', 'ASC-H', 'ASC-US', 'AGC-NOS', 'AGC']
w < 100 and h < 100
small cluster cell nums:20282 clsname: ['LSIL', 'HSIL', 'ASC-US', 'ASC-H', 'AGC-FN', 'AGC-NOS', 'AGC-N', 'AGC']
'''