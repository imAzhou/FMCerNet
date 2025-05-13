'''
1. 将 宽高 < 15 的 anno 清除
2. 将阳性框内部的小框清除
3. 将内含阳性框的阴性框清除
4. 类别映射至：NILM、AGC、ASC-US、LSIL、ASC-H、HSIL, 其余类别丢弃
'''
import uuid
import json
import os
import pandas as pd
from tqdm import tqdm
import copy
from PIL import Image
from cerwsi.utils import KFBSlide,is_bbox_inside
from scripts.data_process_0511.utils import (
    imgName2patientId,box_enclosured,process_noparent_ann,draw_roi_inWSI)

NEGATIVE_CLASS = ['NILM', 'GEC']
POSITIVE_CLASS = ['AGC', 'ASC-US','LSIL', 'ASC-H', 'HSIL']
RECORD_CLASS = {
    'NILM': 'NILM',
    'GEC': 'NILM',
    'ASC-US': 'ASC-US',
    'LSIL': 'LSIL',
    'ASC-H': 'ASC-H',
    'HSIL': 'HSIL',
    'SCC': 'HSIL',
    'AGC-N': 'AGC',
    'AGC': 'AGC',
    'AGC-NOS': 'AGC',
    'AGC-FN': 'AGC',
}
SAFE_MARGIN = 100

def filter_zheyi_annitems(patientId, annitems, max_xy=None):
    rect_items, roi_items = [],[]
    for annitem in annitems:
        if annitem['type'] == 'circle':
            continue
        sub_class = annitem['label']
        all_x,all_y = [p[0] for p in annitem['points']],[p[1] for p in annitem['points']]
        x1,x2 = min(all_x),max(all_x)
        y1,y2 = min(all_y),max(all_y)
        if max_xy is not None:
            x2,y2 = min(max_xy[0], x2), min(max_xy[1], y2)
        bbox_coords = [x1, y1, x2, y2]
        width,height = x2-x1,y2-y1
        new_iteminfo = dict(
            annid = int(str(uuid.uuid4().int)[:13]),      # 取 UUID 转换成的整数的前 13 位
            sub_class=sub_class, region=bbox_coords,
            parent_id = -1
        )
        if sub_class in POSITIVE_CLASS and width>15 and height>15:
            rect_items.append(new_iteminfo)
        if sub_class == 'RoI' and width>15 and height>15:
            roi_items.append(new_iteminfo)
    
    all_boxes = [i['region'] for i in rect_items]
    rect_items = [rect for rect in rect_items if not box_enclosured(rect['region'], all_boxes)]
    rect_has_parent = [False] * len(rect_items)

    for roiitem in roi_items:
        roiitem['children'] = []
        for ridx, annitem in enumerate(rect_items):
            if is_bbox_inside(annitem['region'],roiitem['region'],tolerance=20):
                annitem['parent_id'] = roiitem['annid']
                rect_has_parent[ridx] = True
                roiitem['children'].append(copy.deepcopy(annitem))
    # if patientId == 'ZY_ONLINE_1_8':
    #     print()
    # 为 parent_id == -1 的 annitem 生成 RoI or forge_RoI
    roi_type = 'RoI' if len(roi_items) == 0 else 'forge_RoI'
    noparent_idx = [ridx for ridx,rect in enumerate(rect_items) if rect['parent_id'] == -1]
    if len(noparent_idx) > 0:
        noparent_items = [rect_items[i] for i in range(len(rect_items)) if i in noparent_idx]
        parent_rois = process_noparent_ann(noparent_items, roi_type)
        for tempidx, roiitem in enumerate(parent_rois):
            # if tempidx == 18:
            #     print()
            roiitem['children'] = []
            rx1,ry1,rx2,ry2 = roiitem['region']
            rx1,ry1 = max(0, rx1), max(0, ry1)
            if max_xy is not None:
                rx2,ry2 = min(max_xy[0], rx2), min(max_xy[1], ry2)
            roiitem['region'] = [rx1,ry1,rx2,ry2]
            for ridx, annitem in enumerate(rect_items):
                if is_bbox_inside(annitem['region'],roiitem['region'],tolerance=20):
                    annitem['parent_id'] = roiitem['annid']
                    rect_has_parent[ridx] = True
                    roiitem['children'].append(copy.deepcopy(annitem))
        roi_items.extend(parent_rois)
    
    if sum(rect_has_parent) != len(rect_items):
        print('Retain rect has no parent!')

    return roi_items

def gene_zheyislide_filter():
    '''
    情况1：阳性标注框都在 RoI 内
    情况2：整个WSI没有绘制 RoI，则未标注阳性框的区域都是阴性
    情况3：WSI 有 RoI，RoI外也有部分阳性框
    '''
    df_data_0409 = pd.read_csv('data_resource/zheyi_annofiles/0409_slide_anno.csv')
    df_data_0422 = pd.read_csv('data_resource/zheyi_annofiles/0422_slide_anno.csv')
    df_data = pd.concat([df_data_0409, df_data_0422])
    name2PID = imgName2patientId(df_data)

    with open('data_resource/zheyi_annofiles/宫颈液基细胞—Slide-0409.json', 'r', encoding='utf-8') as f:
        json_data_0409 = json.load(f)
    with open('data_resource/zheyi_annofiles/宫颈液基细胞—Slide-0422.json', 'r', encoding='utf-8') as f:
        json_data_0422 = json.load(f)
    
    slide_items = []
    for group_items in json_data_0409.values():
        slide_items.extend(group_items)
    for group_items in json_data_0422.values():
        slide_items.extend(group_items)

    slide_filter_items = []
    for item in tqdm(slide_items, ncols=80):
        if item['imageName'] == 'K2024-85084_KFBIO_2024-08-01_HSIL.kfb':
            continue
        patientId = name2PID[item['imageName']]
        rowInfo = df_data[df_data['patientId'] == patientId].iloc[0]
        slide = KFBSlide(rowInfo['kfb_path'])
        swidth, sheight = slide.level_dimensions[0]
        filter_anns = filter_zheyi_annitems(patientId, item['annotations'][0]['annotationResult'], 
                                            (swidth-SAFE_MARGIN, sheight-SAFE_MARGIN))
        slide_filter_items.append({
            'patientId': patientId,
            'media_type': 'slide',
            'source_path': rowInfo['kfb_path'],
            'annotations': filter_anns
        })
    with open('data_resource/0511/ann_jsons/zheyi_slide.json', 'w', encoding='utf-8') as f:
        json.dump(slide_filter_items, f, ensure_ascii=False)
    
    for slideitem in tqdm(slide_filter_items, ncols=80):
        slide = KFBSlide(slideitem['source_path'])
        roi_items = [item for item in slideitem['annotations'] if 'RoI' in item['sub_class']]
        save_dir = 'statistic_results/0511/roiInWSI'
        os.makedirs(save_dir, exist_ok=True)
        save_path = f'{save_dir}/{slideitem["patientId"]}.png'
        draw_roi_inWSI(roi_items, slide, save_path)

def gene_zheyiroi_filter():
    df_abandon = pd.read_csv('data_resource/group_csv/abandon.csv')
    with open('data_resource/zheyi_annofiles/宫颈液基细胞—RoI.json', 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    roi_items = []
    group_path = {
        'Group1': '/nfs5/zly/codes/CerWSI/data_resource/zheyi_annofiles/group1/img/',
        'Group3': '/nfs5/zly/codes/CerWSI/data_resource/zheyi_annofiles/group3/img/',
        'Group4': '/nfs5/zly/codes/CerWSI/data_resource/zheyi_annofiles/group4/img/',
    }
    for gname, group_items in json_data.items():
        for item in group_items:
            item['image_path'] = group_path[gname] + item['imageName']
            roi_items.append(item)

    roi_filter_items = []
    for item in tqdm(roi_items, ncols=80):
        patientId = '_'.join(item['imageName'].split('_')[:3])
        abandon_row = df_abandon[df_abandon['patientId'] == patientId]
        if not abandon_row.empty:
            continue
        img = Image.open(item['image_path'])
        w,h = img.size
        annitems = item['annotations'][0]['annotationResult']
        annitems.append(dict(
            id = int(str(uuid.uuid4().int)[:13]),      # 取 UUID 转换成的整数的前 13 位
            type = 'rect', shape = 'rect', label='RoI', 
            points=[[0,0], [w,0], [w,h], [0,h]],
        ))
        filter_anns = filter_zheyi_annitems(patientId, annitems)
        if len(filter_anns) > 1:
            print(f'Error patientId: {patientId} !')
        roi_filter_items.append({
            'patientId': patientId,
            'media_type': 'roi',
            'source_path': item['image_path'],
            'annotations': filter_anns
        })
    with open('data_resource/0511/ann_jsons/zheyi_roi.json', 'w', encoding='utf-8') as f:
        json.dump(roi_filter_items, f, ensure_ascii=False)

def roiImage2mask(imgpath, inside_items):
    pass

def gene_total_pos_annfile():
    df_data = pd.read_csv('data_resource/0511/0_total_pos.csv')

def gene_partial_pos_annfile():
    pass


if __name__ == "__main__":
    # df_abandon = pd.read_csv('data_resource/group_csv/abandon.csv')
    gene_zheyislide_filter()  
    # gene_zheyiroi_filter()

