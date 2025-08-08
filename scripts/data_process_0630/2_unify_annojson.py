'''
1. 将 宽高 < 20 的 anno 清除
2. 将阳性框内部的小框清除
3. 将内含阳性框的阴性框清除
4. 类别映射至：NILM、AGC、ASC-US、LSIL、ASC-H、HSIL, 其余类别丢弃
5. 存储的位置都是相对于 source_path 的坐标，例如 roi image 或者 kfb slide 的左上角(0,0) 坐标
'''
import uuid
import json
import os
from datetime import datetime
import random
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import copy
from PIL import Image
from cerwsi.utils import KFBSlide,is_bbox_inside,decode_xml,read_json_anno,remap_points
from scripts.data_process_0511.utils import (deduplicate_regions,
    imgName2patientId,box_enclosured,process_noparent_ann,draw_roi_inWSI, adjust_region4RoI)

NEGATIVE_CLASS = ['NILM', 'GEC']
POSITIVE_CLASS = ['AGC', 'ASC-US','LSIL', 'ASC-H', 'HSIL', 'SCC']
RECORD_CLASS = {
    'NILM': 'NILM',
    'GEC': 'NILM',
    'ASC-US': 'ASC-US',
    'LSIL': 'LSIL',
    'ASC-H': 'ASC-H',
    'HSIL': 'HSIL',
    'SCC': 'SCC',
    'AGC-N': 'AGC',
    'AGC': 'AGC',
    'AGC-NOS': 'AGC',
    'AGC-FN': 'AGC',
}
SAFE_MARGIN = 100
roi_min_size = 1650

def filter_zheyi_annitems(patientId, annitems, max_xy=None, polygon_shift=0):
    rect_items, roi_items = [],[]
    for annitem in annitems:
        shape = annitem['shape']
        if shape == 'point':
            continue
        
        shift = 0
        if shape == 'polygonPath':
            shift = polygon_shift
        if 'label' not in annitem:
            continue
        sub_class = annitem['label']
        all_x,all_y = [p[0]-shift for p in annitem['points']], [p[1]-shift for p in annitem['points']]
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
        if sub_class in POSITIVE_CLASS and width>20 and height>20:
            rect_items.append(new_iteminfo)
        if sub_class == 'RoI' and width>50 and height>50:
            roi_items.append(new_iteminfo)
    
    rect_items = deduplicate_regions(rect_items)
    # all_boxes = [i['region'] for i in rect_items]
    # rect_items = [rect for rect in rect_items if not box_enclosured(rect['region'], all_boxes)]
    rect_has_parent = [False] * len(rect_items)

    for roiitem in roi_items:
        roiitem['children'] = []
        for ridx, annitem in enumerate(rect_items):
            if is_bbox_inside(annitem['region'],roiitem['region'],tolerance=20):
                annitem['parent_id'] = roiitem['annid']
                rect_has_parent[ridx] = True
                roiitem['children'].append(copy.deepcopy(annitem))
        if len(roiitem['children']) > 0:
            roiitem['region'] = adjust_region4RoI(roiitem['region'], roiitem['children'])

    # 为 parent_id == -1 的 annitem 生成 RoI or forge_RoI
    roi_type = 'RoI' if len(roi_items) == 0 else 'forge_RoI'
    noparent_idx = [ridx for ridx,rect in enumerate(rect_items) if rect['parent_id'] == -1]
    if len(noparent_idx) > 0:
        noparent_items = [rect_items[i] for i in range(len(rect_items)) if i in noparent_idx]
        parent_rois = process_noparent_ann(noparent_items, roi_type, roi_min_size=roi_min_size, max_xy=max_xy, sq_size=500)
        for tempidx, roiitem in enumerate(parent_rois):
            roiitem['children'] = []
            for ridx, annitem in enumerate(rect_items):
                if is_bbox_inside(annitem['region'],roiitem['region'],tolerance=20):
                    annitem['parent_id'] = roiitem['annid']
                    rect_has_parent[ridx] = True
                    roiitem['children'].append(copy.deepcopy(annitem))
            roiitem['region'] = adjust_region4RoI(roiitem['region'], roiitem['children'])
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
    df_data_0607 = pd.read_csv('data_resource/zheyi_annofiles/0607_slide_anno.csv')
    df_abandon = pd.read_csv('data_resource/group_csv/abandon.csv')
    df_data = pd.concat([df_data_0409, df_data_0422, df_data_0607])

    name2PID = imgName2patientId(df_data)

    slide_filter_items = []
    for timetag in ['0409', '0422', '0607']:
        with open(f'data_resource/zheyi_annofiles/宫颈液基细胞—Slide-{timetag}.json', 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        if timetag == '0607':
            del json_data['NILM'] # 暂时不处理阴性 slide 的标注信息
        with open(f'data_resource/zheyi_annofiles/{timetag}_shift.json', 'r', encoding='utf-8') as f:
            shift_config = json.load(f)
        
        slideinfos = []
        for i in json_data.values():
            slideinfos.extend(i)
        for item in tqdm(slideinfos, ncols=90, desc=f'Process timetag: {timetag}'):
            patientId = name2PID[item['imageName']]
            if patientId != '' and patientId not in df_abandon['patientId']:
                rowInfo = df_data[df_data['patientId'] == patientId].iloc[0]
                slide = KFBSlide(rowInfo['kfb_path'])
                swidth, sheight = slide.level_dimensions[0]
                max_xy = (swidth-SAFE_MARGIN, sheight-SAFE_MARGIN)

                item['annotations'] = sorted(
                    item['annotations'],
                    key=lambda d: datetime.strptime(d['updatedTime'], "%Y-%m-%dT%H:%M:%SZ"),
                    reverse=True
                )
                annlist = item['annotations'][0]['annotationResult']
                polygon_shift = shift_config[patientId]
                filter_anns = filter_zheyi_annitems(patientId, annlist, max_xy, polygon_shift)
                slide_filter_items.append({
                    'patientId': patientId,
                    'media_type': 'slide',
                    'source_path': rowInfo['kfb_path'],
                    'annotations': filter_anns
                })
    with open('data_resource/0630/zheyi_slide.json', 'w', encoding='utf-8') as f:
        json.dump(slide_filter_items, f, ensure_ascii=False)
    

def gene_zheyiroi_filter():
    df_abandon = pd.read_csv('data_resource/group_csv/abandon.csv')
    with open('data_resource/zheyi_annofiles/宫颈液基细胞—RoI.json', 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    roi_items = []
    group_path = {
        'Group1': '/nfs5/zly/codes/CerWSI/data_resource/zheyi_annofiles/group1/img/',
        'Group2': '/nfs5/zly/codes/CerWSI/data_resource/zheyi_annofiles/group2/img/',
        'Group3': '/nfs5/zly/codes/CerWSI/data_resource/zheyi_annofiles/group3/img/',
        'Group4': '/nfs5/zly/codes/CerWSI/data_resource/zheyi_annofiles/group4/img/',
    }
    for gname, group_items in json_data.items():
        for item in group_items:
            item['image_path'] = group_path[gname] + item['imageName']
            roi_items.append(item)

    roi_filter_items = []
    for item in tqdm(roi_items, ncols=90, desc=f'Process: zheyi roi'):
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
    with open('data_resource/0630/zheyi_roi.json', 'w', encoding='utf-8') as f:
        json.dump(roi_filter_items, f, ensure_ascii=False)

def gene_jfsw_slide_filter():
    df_data = pd.read_csv('data_resource/0630/5_jfsw_train.csv')
    df_data = df_data[df_data['kfb_clsid'] == 1]
    slide_filter_items = []
    for rowInfo in tqdm(df_data.itertuples(index=False), total=len(df_data),ncols=90, desc=f'Process: JFSW slide'):
        # if rowInfo.patientId != 'JFSW_2_2421':
        #     continue
        slide = KFBSlide(rowInfo.kfb_path)
        swidth, sheight = slide.level_dimensions[0]
        max_xy = (swidth-SAFE_MARGIN, sheight-SAFE_MARGIN)
        annos = read_json_anno(rowInfo.anno_path)
        rect_items = []
        for ann_ in annos:
            ann = remap_points(ann_)
            if ann is None:
                continue
            if ann.get('sub_class') not in RECORD_CLASS.keys():
                continue
            sub_class = RECORD_CLASS[ann.get('sub_class')]
            region = ann.get('region')
            x,y = region['x'],region['y']
            w,h = region['width'],region['height']

            if w <=20 or h<=20 or sub_class not in POSITIVE_CLASS:
                continue
            new_iteminfo = dict(
                annid = int(str(uuid.uuid4().int)[:13]),
                sub_class=sub_class, region=[x,y,x+w,y+h],
                parent_id = -1
            )
            rect_items.append(new_iteminfo)
        
        rect_items = deduplicate_regions(rect_items)
        all_boxes = [i['region'] for i in rect_items]
        rect_items = [rect for rect in rect_items if not box_enclosured(rect['region'], all_boxes)]
        rect_has_parent = [False] * len(rect_items)
        parent_rois = process_noparent_ann(rect_items, 'forge_RoI', roi_min_size=roi_min_size, max_xy=max_xy, sq_size=500)
        for tempidx, roiitem in enumerate(parent_rois):
            roiitem['children'] = []
            for ridx, annitem in enumerate(rect_items):
                if is_bbox_inside(annitem['region'],roiitem['region'],tolerance=20):
                    annitem['parent_id'] = roiitem['annid']
                    rect_has_parent[ridx] = True
                    roiitem['children'].append(copy.deepcopy(annitem))
            roiitem['region'] = adjust_region4RoI(roiitem['region'], roiitem['children'])

        if sum(rect_has_parent) != len(rect_items):
            print('Retain rect in JFSW has no parent!')
        
        if len(parent_rois) > 0:
            slide_filter_items.append({
                'patientId': rowInfo.patientId,
                'media_type': 'slide',
                'source_path': rowInfo.kfb_path,
                'annotations': parent_rois
            })
        else:
            print(f'Empty in jfsw_pos: {rowInfo.patientId}')
    with open('data_resource/0630/jfsw_pos_slide.json', 'w', encoding='utf-8') as f:
        json.dump(slide_filter_items, f, ensure_ascii=False)

def gene_wxl1_slide_filter():
    df_data = pd.read_csv('data_resource/0630/0_zheyi_pos.csv')
    df_data = df_data[(df_data['anno_type'] == 'partial') & (df_data['kfb_source'] == 'WXL_1')]
    slide_filter_items = []
    for rowInfo in tqdm(df_data.itertuples(index=False), total=len(df_data), ncols=90, desc=f'Process: WXL1 slide'):

        slide = KFBSlide(rowInfo.kfb_path)
        swidth, sheight = slide.level_dimensions[0]
        max_xy = (swidth-SAFE_MARGIN, sheight-SAFE_MARGIN)
        all_rects = decode_xml(rowInfo.anno_path)
        rect_items = []
        for x1,y1,x2,y2 in all_rects:
            w,h = x2-x1, y2-y1
            sub_class = RECORD_CLASS[rowInfo.kfb_clsname]
            if w <=20 or h<=20 or sub_class not in POSITIVE_CLASS:
                continue
            new_iteminfo = dict(
                annid = int(str(uuid.uuid4().int)[:13]),
                sub_class=sub_class, region=[x1,y1,x2,y2],
                parent_id = -1
            )
            rect_items.append(new_iteminfo)
        rect_items = deduplicate_regions(rect_items)
        # all_boxes = [i['region'] for i in rect_items]
        # rect_items = [rect for rect in rect_items if not box_enclosured(rect['region'], all_boxes)]
        rect_has_parent = [False] * len(rect_items)
        parent_rois = process_noparent_ann(rect_items, 'forge_RoI', roi_min_size=roi_min_size, max_xy=max_xy, sq_size=500)
        for tempidx, roiitem in enumerate(parent_rois):
            roiitem['children'] = []
            for ridx, annitem in enumerate(rect_items):
                if is_bbox_inside(annitem['region'],roiitem['region'],tolerance=20):
                    annitem['parent_id'] = roiitem['annid']
                    rect_has_parent[ridx] = True
                    roiitem['children'].append(copy.deepcopy(annitem))
            roiitem['region'] = adjust_region4RoI(roiitem['region'], roiitem['children'])
        if sum(rect_has_parent) != len(rect_items):
            print('Retain rect in WXL has no parent!')

        slide_filter_items.append({
            'patientId': rowInfo.patientId,
            'media_type': 'slide',
            'source_path': rowInfo.kfb_path,
            'annotations': parent_rois
        })
    with open('data_resource/0630/wxl_pos_slide.json', 'w', encoding='utf-8') as f:
        json.dump(slide_filter_items, f, ensure_ascii=False)

def reset_trainval_pids(all_json_datas):
    '''
    Val: 阳性 pid 去重后300，阴性 pid 300
    Val 阳性 pid: 含阳性病变的 RoI 所在 pid，20 张浙一病理slide
    '''
    df_pure_trainval = pd.read_csv('data_resource/0630/4_pure_trainval.csv')
    df_neg = df_pure_trainval[df_pure_trainval['kfb_clsid'] == 0]
    neg_pids = list(df_neg['patientId'])
    random.shuffle(neg_pids)
    
    include_lesion_roi_pids = []
    for idx, item in enumerate(tqdm(all_json_datas, ncols=80)):
        for RoIItem in item['annotations']:
            if len(RoIItem['children']) > 0 and RoIItem['sub_class'] == 'RoI':
                include_lesion_roi_pids.append(item['patientId'])
    unique_pids = list(set(include_lesion_roi_pids))
    random.shuffle(unique_pids)
    zheyi_slide = [
        'WXL_1_94', 'ZY_ONLINE_1_170',    # SCC
        'ZY_ONLINE_1_135','ZY_ONLINE_1_104', 'ZY_ONLINE_1_198', 'ZY_ONLINE_1_1481', 'WXL_1_113',   # HSIL
        'ZY_ONLINE_1_1479', 'ZY_ONLINE_1_21', 'ZY_ONLINE_1_43', 'ZY_ONLINE_1_1450', 'WXL_1_178',    # LSIL
        'ZY_ONLINE_1_14', 'ZY_ONLINE_1_8', 'ZY_ONLINE_1_1442', 'WXL_1_331',      # ASC-US
        'ZY_ONLINE_1_101', 'ZY_ONLINE_1_65', 'ZY_ONLINE_1_1469', 'WXL_1_215',   # ASC-H
    ]
    filter_unique_pids = [i for i in unique_pids if i not in zheyi_slide]
    val_pos_pids = [*filter_unique_pids[:290], *zheyi_slide]
    val_neg_pids = neg_pids[:300]
    val_pids = [*val_pos_pids, *val_neg_pids]   # Pos310 / Neg300
    
    df_new_val = df_pure_trainval[df_pure_trainval['patientId'].isin(val_pids)]
    df_new_pure_train = df_pure_trainval[~df_pure_trainval['patientId'].isin(df_new_val['patientId'])]

    df_new_val.to_csv('data_resource/0630/6_val.csv', index=False)
    df_new_pure_train.to_csv('data_resource/0630/4_pure_train.csv', index=False)

def statistic_pids(all_json_datas, jfsw_pos_slide):

    error_flag = False
    json_pid_nums = len(set([i['patientId'] for i in all_json_datas]))
    df_pure_train = pd.read_csv('data_resource/0630/4_pure_train.csv')
    df_pure_train_pos = df_pure_train[df_pure_train['kfb_clsid']==1]
    df_val = pd.read_csv('data_resource/0630/6_val.csv')
    df_val_pos = df_val[df_val['kfb_clsid']==1]
    csv_pids = [*list(set(df_pure_train_pos['patientId'])), *list(set(df_val_pos['patientId']))]
    if json_pid_nums != len(csv_pids):
        print('ERROR: pure_train + val pids not right.')
        error_flag = True

    df_jfsw_train = pd.read_csv('data_resource/0630/5_jfsw_train.csv')
    df_jfsw_train_pos = df_jfsw_train[df_jfsw_train['kfb_clsid']==1]
    if len(jfsw_pos_slide) != len(df_jfsw_train_pos):
        print('ERROR: jfsw_train pids not right.')
        error_flag = True
    
    if not error_flag:
        print('Data Clean!')
        reload_patchlist = defaultdict(list)
        for patchinfo in tqdm(all_json_datas, ncols=80):
            keyname = f"{patchinfo['patientId']}_{patchinfo['media_type']}"
            reload_patchlist[keyname].append(patchinfo)
        print(f'Unique pid nums in pure_tran and val: {json_pid_nums}')
        print(f'Unique pid_mediatype nums in pure_tran and val: {len(reload_patchlist)}')
        print(f'Unique pid_mediatype nums in jfsw_tran: {len(jfsw_pos_slide)}')

        reload_pidlist = defaultdict(list)
        for patchinfo in tqdm(all_json_datas, ncols=80):
            reload_pidlist[patchinfo['patientId']].append(patchinfo)
        
        repeat_pids = []
        for pid,children in reload_pidlist.items():
            media_type = set([i['media_type'] for i in children])
            if len(media_type) > 1:
                repeat_pids.append(pid)
        print(f'Two type medial_type pid nums: {len(repeat_pids)}:')
        # print(repeat_pids)

if __name__ == "__main__":
    '''gene_*() 函数会给每个 RoI/forge_RoI 以及 annitem 重新生成 uuid，若已经生成 roi_mask，
    请慎重执行该函数，会导致 roi_mask npz 文件失去对应的 RoI 信息(两者之间的唯一联系是 uuid)'''
    gene_zheyislide_filter()
    gene_zheyiroi_filter()
    gene_wxl1_slide_filter()
    gene_jfsw_slide_filter()

    # with open('data_resource/0630/zheyi_roi.json', 'r', encoding='utf-8') as f:
    #     zheyi_roi_data = json.load(f)
    # with open('data_resource/0630/zheyi_slide.json', 'r', encoding='utf-8') as f:
    #     zheyi_slide = json.load(f)
    # with open('data_resource/0630/wxl_pos_slide.json', 'r', encoding='utf-8') as f:
    #     wxl_pos_slide = json.load(f)
    # with open('data_resource/0630/jfsw_pos_slide.json', 'r', encoding='utf-8') as f:
    #     jfsw_pos_slide = json.load(f)

    # all_json_datas = [*zheyi_roi_data, *zheyi_slide, *wxl_pos_slide]
    # reset_trainval_pids(all_json_datas)
    # statistic_pids(all_json_datas, jfsw_pos_slide)
    