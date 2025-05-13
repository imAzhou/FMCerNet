import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from cerwsi.utils import KFBSlide,read_json_anno,is_bbox_inside,remap_points

POSITIVE_CLASS = ['ASC-US', 'LSIL', 'ASC-H', 'HSIL', 'SCC', 'AGC-NOS', 'AGC', 'AGC-N', 'AGC-FN']
colors = plt.cm.tab10(np.linspace(0, 1, len(POSITIVE_CLASS)))[:, :3] * 255
category_colors = {cat: tuple(map(int, color)) for cat, color in zip(POSITIVE_CLASS, colors)}

def get_ROI_inside(roi_rect, anns):
    roi_x,roi_y,roi_w,roi_h = roi_rect
    roi_x1y1x2y2 = [roi_x,roi_y,roi_x+roi_w,roi_y+roi_h]
    item_inside = []
    for ann_ in anns:
        ann = remap_points(ann_)
        if ann is None:
            return item_inside
        sub_class = ann.get('sub_class')
        region = ann.get('region')
        x,y = region['x'],region['y']
        w,h = region['width'],region['height']
        if is_bbox_inside([x,y,x+w,y+h], roi_x1y1x2y2, tolerance=5) and sub_class != 'ROI':
            item_inside.append(ann)
    return item_inside

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
    # patches = [
    #     mpatches.Patch(color=np.array(color) / 255.0, label=category)  # Matplotlib 支持归一化颜色
    #     for category, color in category_colors.items()
    # ]
    # ax.legend(handles=patches, loc='upper right', bbox_to_anchor=(1.35, 1), frameon=False)
    # fig.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

def filter_inside(ann_items):
    del_idx = []
    for ann_idx, ann in enumerate(ann_items):
        category = ann.get('sub_class')
        region = ann.get('region')
        x,y = region['x'],region['y']
        w,h = region['width'],region['height']

        for ann_compare in ann_items:
            if ann['id'] == ann_compare['id']:
                continue
            category_ = ann_compare.get('sub_class')
            region_ = ann_compare.get('region')
            x_,y_ = region_['x'],region_['y']
            w_,h_ = region_['width'],region_['height']
            if is_bbox_inside([x,y,x+w,y+h], [x_,y_,x_+w_,y_+h_], tolerance=4) and category == category_:
                del_idx.append(ann_idx)
                break
    result = []            
    for i,item in enumerate(ann_items):
        if i not in del_idx:
            result.append(item)
    return result

if __name__ == '__main__':
    data_root_dir = '/medical-data/data'

    df_jf1 = pd.read_csv('data_resource/cls_pn/group_csv/JFSW_1.csv')
    df_jf2 = pd.read_csv('data_resource/cls_pn/group_csv/JFSW_2.csv')
    df_jf = pd.concat([df_jf1, df_jf2], ignore_index=True)
    
    ASC_CLASS = ['ASC-US', 'LSIL', 'ASC-H', 'HSIL', 'SCC']
    
    for row in tqdm(df_jf.itertuples(index=True), total=len(df_jf), ncols=80):

        if not isinstance(row.json_path, str) or 'AGC' not in row.kfb_clsname:
            continue
        json_path = f'{data_root_dir}/{row.json_path}'
        annos = read_json_anno(json_path)
        save_nums = 0
        for ann in annos:
            sub_class = ann.get('sub_class')
            if sub_class != 'ROI':
                continue
            roi_item = remap_points(ann)
            if roi_item is None:
                continue
            region = roi_item.get('region')
            x,y = region['x'],region['y']
            w,h = region['width'],region['height']
            if w > 3000 and h > 3000:
                continue
            ROI_inside = get_ROI_inside([x,y,w,h], annos)
            ROI_inside = filter_inside(ROI_inside)
            ASC_nums = sum([1 if i.get('sub_class') in ASC_CLASS else 0 for i in ROI_inside])
            if ASC_nums > 1:
                slide = KFBSlide(f'{data_root_dir}/{row.kfb_path}')
                location, level, size = (x,y), 0, (w,h)
                read_result = Image.fromarray(slide.read_region(location, level, size))
                save_path = f'statistic_results/AGC_check/{row.patientId}.png'
                draw_OD(read_result, save_path, [x,y,w,h], ROI_inside)
                save_nums += 1
                break
        if save_nums == 10:
            break

