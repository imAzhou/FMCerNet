import json
import os
from tqdm import tqdm
from cerwsi.utils import KFBSlide,read_json_anno,is_bbox_inside
import pandas as pd
import random
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

NEGATIVE_CLASS = ['NILM', 'GEC']
POSITIVE_CLASS = ['ASC-US', 'LSIL', 'ASC-H', 'HSIL', 'SCC', 'AGC-NOS', 'AGC', 'AGC-N', 'AGC-FN']
WINDOW_SIZE = 500
RANDOM_NUM = 2
VISUAL_INSIDE = True

def random_square(x1,y1,w,h):
    if w == WINDOW_SIZE:
        w += 0.1
    if h == WINDOW_SIZE:
        h += 0.1
    if w<WINDOW_SIZE and h<WINDOW_SIZE:
        rest_w, rest_h = WINDOW_SIZE-w, WINDOW_SIZE-h
        interval_x = [x1-rest_w, x1]
        interval_y = [y1-rest_h, y1]
    elif w>WINDOW_SIZE and h>WINDOW_SIZE:
        interval_x = [x1, x1+(w-WINDOW_SIZE)]
        interval_y = [y1, y1+(h-WINDOW_SIZE)]
    elif w>WINDOW_SIZE and h<WINDOW_SIZE:
        interval_x = [x1, x1+(w-WINDOW_SIZE)]
        rest_h = WINDOW_SIZE-h
        interval_y = [y1-rest_h, y1]
    elif w<WINDOW_SIZE and h>=WINDOW_SIZE:
        interval_y = [y1, y1+(h-WINDOW_SIZE)]
        rest_w = WINDOW_SIZE-w
        interval_x = [x1-rest_w, x1]
    else:
        print(f'x1:{x1},y1:{y1},w:{w},h:{h} Not Matched!')
    
    cut_results = []
    for _ in range(RANDOM_NUM):
        new_x,new_y = random.randint(int(interval_x[0]),int(interval_x[1])),random.randint(int(interval_y[0]),int(interval_y[1]))
        cut_results.append([new_x,new_y,WINDOW_SIZE,WINDOW_SIZE])
    return cut_results

def check_cls_inside(square_coord, kfb_anno, original_coord):
    square_x1,square_y1,square_w,square_h = square_coord
    coords = [square_x1, square_y1, square_x1+square_w, square_y1+square_h]
    cls_inside = []
    for idx,i in enumerate(kfb_anno):
        region = i.get('region')
        sub_class = i.get('sub_class')
        w,h = region['width'],region['height']
        x1,y1 = region['x'],region['y']
        x2,y2 = x1+w, y1+h
        o_x1,o_y1,o_x2,o_y2 = original_coord
        if w<0 or h<0:
            continue
        if x1 == o_x1 and y1 == o_y1 and x2 == o_x2 and y2 == o_y2:
            cls_inside.append(dict(clsname=sub_class, coords=[x1,y1,x2,y2]))
        elif is_bbox_inside([x1,y1,x2,y2], coords) and sub_class in POSITIVE_CLASS:
            cls_inside.append(dict(clsname=sub_class, coords=[x1,y1,x2,y2]))
    return cls_inside

def draw_OD(read_image, save_path, square_coords, inside_items):
    draw = ImageDraw.Draw(read_image)
    sq_x1,sq_y1,sq_w,sq_h = square_coords

    for box_item in inside_items:
        category = box_item['clsname']
        x1, y1, x2, y2 = box_item['coords']
        x_min = max(sq_x1, x1) - sq_x1
        y_min = max(sq_y1, y1) - sq_y1
        x_max = min(sq_x1+sq_w, x2) - sq_x1
        y_max = min(sq_y1+sq_h, y2) - sq_y1
        
        color = category_colors.get(category, (255, 255, 255))
        draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=3)
        draw.text((x_min + 2, y_min - 15), category, fill=color)
    
    # 使用 matplotlib 添加 legend
    fig, ax = plt.subplots()
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

def cut_patch(valid_imgs, mode):
    save_dir = 'data_resource/cls_pn/cut_img'
    square_save_path = f'{save_dir}/square_cut'
    os.makedirs(square_save_path, exist_ok=True)

    if VISUAL_INSIDE:
        inside_save_path = f'{save_dir}/square_cut_OD'
        os.makedirs(inside_save_path, exist_ok=True)

    mode_sq_data = []
    for slide_valid_item in tqdm(valid_imgs, ncols=80):
        patientId = slide_valid_item['patientId']
        kfb_clsname = slide_valid_item['kfb_clsname']

        if 'JFSW' not in patientId or kfb_clsname == 'NILM':
            continue
        filtered = df_JF.loc[df_JF['patientId'] == patientId]
        if filtered.empty:
            print("No matching patientId found.")
            continue
        
        slide = KFBSlide(slide_valid_item['kfb_path'])
        row = filtered.iloc[0]
        kfb_anno = read_json_anno(row.json_path)
        for idx, img_item in enumerate(slide_valid_item['valid_anno']):
            patch_clsname = img_item['patch_clsname']
            x1,y1,x2,y2 = img_item['coord']
            w,h = img_item['size']

            if patch_clsname not in NEGATIVE_CLASS:
                cut_results = random_square(x1,y1,w,h)
                for j,new_rect in enumerate(cut_results):
                    new_x1,new_y1,new_w,new_h = new_rect
                    location, level, size = (new_x1,new_y1), 0, (new_w,new_h)
                    read_result = Image.fromarray(slide.read_region(location, level, size))
                    filename = f'{patientId}_sq{idx}{j}.png'
                    read_result.save(f'{square_save_path}/{filename}')
                    cls_inside = check_cls_inside(new_rect, kfb_anno, img_item['coord'])
                    cls_names = [item['clsname'] for item in cls_inside]
                    mode_sq_data.append([filename, ','.join(cls_names)])
                    if VISUAL_INSIDE:
                        save_path = f'{inside_save_path}/{filename}'
                        draw_OD(read_result, save_path, new_rect,cls_inside)
    
    
    df_mode_sq = pd.DataFrame(mode_sq_data, columns=['filename', 'inside_clsname'])
    df_mode_sq.to_csv(f'{save_dir}/sqare_{mode}.csv', index=False)
    

if __name__ == '__main__':
    with open('data_resource/cls_pn/1127_anno_train.json','r') as f:
        train_data = json.load(f)
    with open('data_resource/cls_pn/1127_anno_val.json','r') as f:
        val_data = json.load(f)

    df_JF_1 = pd.read_csv('data_resource/cls_pn/group_csv/JFSW_1.csv')
    df_JF_2 = pd.read_csv('data_resource/cls_pn/group_csv/JFSW_2.csv')
    df_JF = pd.concat([df_JF_1, df_JF_2], ignore_index=True)

    colors = plt.cm.tab10(np.linspace(0, 1, len(POSITIVE_CLASS)))[:, :3] * 255
    category_colors = {cat: tuple(map(int, color)) for cat, color in zip(POSITIVE_CLASS, colors)}
    
    cut_patch(train_data['valid_imgs'], 'train')
    cut_patch(val_data['valid_imgs'], 'val')
