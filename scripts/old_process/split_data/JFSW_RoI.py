import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import json
import glob
import random
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from cerwsi.utils import KFBSlide,read_json_anno,is_bbox_inside,random_cut_square,remap_points,generate_cut_regions

POSITIVE_CLASS = ['ASC-US', 'LSIL', 'ASC-H', 'HSIL', 'SCC', 'AGC-NOS', 'AGC', 'AGC-N', 'AGC-FN']
colors = plt.cm.tab10(np.linspace(0, 1, len(POSITIVE_CLASS)))[:, :3] * 255
category_colors = {cat: tuple(map(int, color)) for cat, color in zip(POSITIVE_CLASS, colors)}


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


def get_cutregion_inside(square_coord, inside_items):
    '''
    Args:
        square_coord: (x1, y1, x2, y2)
    '''
    item_inside = []
    for idx,i in enumerate(inside_items):
        region = i.get('region')
        w,h = region['width'],region['height']
        x1,y1 = region['x'],region['y']
        x2,y2 = x1+w, y1+h
        if w<0 or h<0:
            continue
        if is_bbox_inside([x1,y1,x2,y2], square_coord, tolerance=5):
            item_inside.append(dict(id=i['id'], sub_class=i['sub_class'], region=region))
    return item_inside

def get_ROI_inside(roi_rect, anns):
    roi_x,roi_y,roi_w,roi_h = roi_rect
    roi_x1y1x2y2 = [roi_x,roi_y,roi_x+roi_w,roi_y+roi_h]
    item_inside = []
    for ann_ in anns:
        ann = remap_points(ann_)
        if ann is None:
            return item_inside
        region = ann.get('region')
        sub_class = ann.get('sub_class')
        x,y = region['x'],region['y']
        w,h = region['width'],region['height']
        if w>100 and h>100 and is_bbox_inside([x,y,x+w,y+h], roi_x1y1x2y2, tolerance=5) and sub_class != 'ROI':
            item_inside.append(ann)
    return item_inside

def square_cut_roi(roi_rect, ROI_inside):
    x,y,w,h = roi_rect
    roi_item_dict = dict(xywh=roi_rect, cut_x1y1=[], ann_in_square=[])

    cut_points = generate_cut_regions((x,y), w, h, WINDOW_SIZE)
    check_inside_item = []
    for rect_coords in cut_points:
        x1,y1 = rect_coords
        item_inside = get_cutregion_inside([x1,y1,x1+WINDOW_SIZE,y1+WINDOW_SIZE], ROI_inside)
        check_inside_item.extend(item_inside)
        roi_item_dict['cut_x1y1'].append([x1,y1])
        roi_item_dict['ann_in_square'].append(item_inside)
    unique_itemid = set([i['id'] for i in check_inside_item])
    if len(unique_itemid) != len(ROI_inside):
        unassigned = [item for item in ROI_inside if item['id'] not in unique_itemid]
        for una_ann in unassigned:
            ur = una_ann.get('region')
            x1,y1 = random_cut_square([ur['x'], ur['y'],ur['width'],ur['height']], WINDOW_SIZE)
            roi_item_dict['cut_x1y1'].append([x1,y1])
            item_inside = get_cutregion_inside([x1,y1,x1+WINDOW_SIZE,y1+WINDOW_SIZE], ROI_inside)
            if ur['width'] > WINDOW_SIZE or ur['height'] > WINDOW_SIZE:
                crop_region = dict(
                    x = min(ur['x'], x1),
                    y = min(ur['y'], y1),
                    width = min(ur['width'], WINDOW_SIZE),
                    height = min(ur['height'], WINDOW_SIZE),
                )
                item_inside.append(
                    dict(id=una_ann['id'], sub_class=una_ann['sub_class'], region=crop_region)
                )
            roi_item_dict['ann_in_square'].append(item_inside)
    return roi_item_dict

def get_RoI_info():
    df_jf1 = pd.read_csv('data_resource/cls_pn/group_csv/JFSW_1.csv')
    df_jf2 = pd.read_csv('data_resource/cls_pn/group_csv/JFSW_2.csv')
    df_jf = pd.concat([df_jf1, df_jf2], ignore_index=True)
    
    no_roi_list = []
    record_kfb = [0,0]
    for row in tqdm(df_jf.itertuples(index=True), total=len(df_jf), ncols=80):

        if not isinstance(row.json_path, str):
            continue
        if os.path.exists(f'{json_savedir}/{row.patientId}.json'):
            idx = 0 if row.kfb_clsname == 'NILM' else 1
            record_kfb[idx] += 1
            continue
        json_path = f'{data_root_dir}/{row.json_path}'
        annos = read_json_anno(json_path)
        filter_roi = dict(roi_list = [], cut_window_size = WINDOW_SIZE)
        ROI_items = [ann for ann in annos if ann['sub_class'] == 'ROI']
        for item_ in ROI_items:
            roi_item = remap_points(item_)
            if roi_item is None:
                continue
            region = roi_item.get('region')
            x,y = region['x'],region['y']
            w,h = region['width'],region['height']
            if w > WINDOW_SIZE and h > WINDOW_SIZE:
                ROI_inside = get_ROI_inside([x,y,w,h], annos)
                if len(ROI_inside) > 0:
                    roi_item_dict = square_cut_roi([x,y,w,h], ROI_inside)
                    filter_roi['roi_list'].append(roi_item_dict)
        
        if len(filter_roi['roi_list']) == 0:
            no_roi_list.append(f'{row.kfb_path}\n')
        else:
            idx = 0 if row.kfb_clsname == 'NILM' else 1
            record_kfb[idx] += 1
            with open(f'{json_savedir}/{row.patientId}.json', 'w') as f:
                json.dump(filter_roi, f)
    
    with open('statistic_results/jfsw_no_roi_v2.txt', 'w') as f:
        f.writelines(no_roi_list)
    
    print(f'total kfb nums: {sum(record_kfb)}, Neg: {record_kfb[0]}, Pos:{record_kfb[1]}')

def draw_squ_item(item, slide, save_filename):
    square_clsid = 0 if item['pos_nums'] == 0 else 1
    square_x1y1 = item['square_x1y1']
    location, level, size = square_x1y1, 0, (WINDOW_SIZE,WINDOW_SIZE)
    read_result = Image.fromarray(slide.read_region(location, level, size))
    read_result.save(f'{img_savedir}/{save_filename}')

    if square_clsid == 1 and random.random() < 0.001:
        save_path = f'{imgOD_savedir}/{save_filename}'
        draw_OD(read_result, save_path, [*square_x1y1, WINDOW_SIZE,WINDOW_SIZE], item['pos_anns'])
    


def cut_RoI_Img():
    df_jf1 = pd.read_csv('data_resource/cls_pn/group_csv/JFSW_1.csv')
    df_jf2 = pd.read_csv('data_resource/cls_pn/group_csv/JFSW_2.csv')
    df_jf = pd.concat([df_jf1, df_jf2], ignore_index=True)
    annojson_files = glob.glob(f'{json_savedir}/*.json')

    retain_pos_slide = dict()
    squ_record = [0,0]
    all_flatten_squ = []
    pos_kfb_data = []
    
    for annojson_f in tqdm(annojson_files, ncols=80):
        patientId = os.path.basename(annojson_f).split('.')[0]
        patient_row = df_jf.loc[df_jf['patientId'] == patientId].iloc[0]
        if patient_row.kfb_clsname == 'NILM':
            continue
        with open(annojson_f, 'r') as f:
            roi_anno = json.load(f)
        
        slide = KFBSlide(f'{data_root_dir}/{patient_row.kfb_path}')
        slide_width, slide_height = slide.level_dimensions[0]
        img_uniid_neg,img_uniid_pos = 0,0
        flatten_squ = []
        pos_square = 0
        for roii, singleroi in enumerate(roi_anno['roi_list']):
            for square_x1y1, anns in zip(singleroi['cut_x1y1'], singleroi['ann_in_square']):
                sx1,sy1,sx2,sy2 = square_x1y1[0], square_x1y1[1], square_x1y1[0]+WINDOW_SIZE, square_x1y1[1]+WINDOW_SIZE
                if sx1 > slide_width or sy1 > slide_height or sx2 > slide_width or sy2 > slide_height:
                    continue

                pos_ann = [ann for ann in anns if ann['sub_class'] in POSITIVE_CLASS]
                flatten_squ.append({
                    'patientId': patientId,
                    'roi_xywh': singleroi['xywh'],
                    'square_x1y1': square_x1y1,
                    'pos_anns': pos_ann,
                    'pos_nums': len(pos_ann),
                    'pos_clsname': [a['sub_class'] for a in pos_ann],
                })
                if len(pos_ann) > 0:
                    pos_square += 1

        if pos_square > 5 or (patient_row.kfb_clsname == 'ASC-US' and pos_square > 3):
            retain_pos_slide[patient_row.kfb_clsname] = retain_pos_slide.get(patient_row.kfb_clsname, 0) + 1
            pos_kfb_data.append(patient_row)

            for item in flatten_squ:
                if item['pos_nums'] == 0 and random.random() < 0.1:
                    save_filename = f'{patientId}_neg{img_uniid_neg}.png'
                    item['filename'] = save_filename
                    all_flatten_squ.append(item)
                    img_uniid_neg += 1
                    # draw_squ_item(item, slide, save_filename)
                elif item['pos_nums'] > 0:
                    save_filename = f'{patientId}_pos{img_uniid_pos}.png'
                    item['filename'] = save_filename
                    all_flatten_squ.append(item)
                    # draw_squ_item(item, slide, save_filename)
                    img_uniid_pos += 1
            
            squ_record[0] += img_uniid_neg
            squ_record[1] += img_uniid_pos
    
    total_slide = 0
    for key,value in retain_pos_slide.items():
        print(f'{key}: {value}')
        total_slide += value
    print(f'total slide: {total_slide}, total square: {sum(squ_record)}, Neg: {squ_record[0]}, Pos: {squ_record[1]}')
    df_pos_kfb_data = pd.DataFrame(pos_kfb_data)
    df_pos_kfb_data.to_csv('data_resource/ROI/annofile/1223_pos_v2.csv', index=False)
    square_in_pos_dict = dict(square_nums={'total':sum(squ_record), 'Neg':squ_record[0], 'Pos':squ_record[1]},
                              square_items=all_flatten_squ)
    with open('data_resource/ROI/annofile/square_in_pos_v2.json', 'w') as f:
        json.dump(square_in_pos_dict, f)
    
    # with open('statistic_results/pos_no_ann.txt', 'w') as f:
    #     f.writelines(pos_no_ann)

def filter_neg_slide():
    with open('data_resource/cls_pn/group_csv/jfsw_NILM_pos_ann.txt', 'r') as f:
        NILM_pos_ann = f.readlines()
    NILM_pos_ann = [i.strip() for i in NILM_pos_ann]
    
    df_jf1 = pd.read_csv('data_resource/cls_pn/group_csv/JFSW_1.csv')
    df_jf2 = pd.read_csv('data_resource/cls_pn/group_csv/JFSW_2.csv')
    df_jf = pd.concat([df_jf1, df_jf2], ignore_index=True)

    df_wxl1 = pd.read_csv('data_resource/cls_pn/group_csv/WXL_1.csv')
    df_wxl2 = pd.read_csv('data_resource/cls_pn/group_csv/WXL_3.csv')
    df_wxl = pd.concat([df_wxl1, df_wxl2], ignore_index=True)

    # kfb_path,kfb_clsid,kfb_clsname,patientId,kfb_source
    neg_kfbinfo = []
    for row in tqdm(df_jf.itertuples(index=True), total=len(df_jf), ncols=80):
        if row.kfb_clsname != 'NILM' or row.kfb_path in NILM_pos_ann:
            continue
        kfb_source = 'JFSW_1' if 'JFSW_1' in row.patientId else 'JFSW_2'
        neg_kfbinfo.append([row.kfb_path, 0, row.kfb_clsname, row.patientId, kfb_source])
    
    for row in tqdm(df_wxl.itertuples(index=True), total=len(df_wxl), ncols=80):
        if row.kfb_clsname != 'NILM':
            continue
        kfb_source = 'WXL_1' if 'WXL_1' in row.patientId else 'WXL_3'
        neg_kfbinfo.append([row.kfb_path, 0, row.kfb_clsname, row.patientId, kfb_source])

    neg_csv = pd.DataFrame(neg_kfbinfo, columns = ['kfb_path','kfb_clsid','kfb_clsname','patientId','kfb_source'])
    neg_csv.to_csv('data_resource/ROI/annofile/1223_neg.csv', index=False)

if __name__ == '__main__':
    data_root_dir = '/medical-data/data'
    json_savedir = 'data_resource/ROI/annojson4roi_v2'
    img_savedir = 'data_resource/ROI/images_v2'
    imgOD_savedir = 'data_resource/ROI/imagesOD_v2'
    os.makedirs(json_savedir, exist_ok=True)
    os.makedirs(img_savedir, exist_ok=True)
    os.makedirs(imgOD_savedir, exist_ok=True)
    WINDOW_SIZE = 500
    # get_RoI_info()
    cut_RoI_Img()
    # filter_neg_slide()


'''
total kfb nums: 2556, Neg: 342, Pos:2214
in pos slide: total square: 1786457, Neg: 1668372, Pos: 118085

total slide: 915, total square: 1000188, Neg: 897880, Pos: 102308
394 - 26 + 100 + 148 + 341 = 957

HSIL: 283
ASC-H: 183
ASC-US: 155
LSIL: 183
AGC: 217
total slide: 1021, total square: 490473, Neg: 386251, Pos: 104222


==== v2 ====
total kfb nums: 2355, Neg: 284, Pos:2071

HSIL: 250
ASC-US: 109
LSIL: 324
AGC: 263
ASC-H: 127
total slide: 1073, total square: 115028, Neg: 93703, Pos: 21325
'''