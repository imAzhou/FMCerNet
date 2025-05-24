import uuid
import os
from tqdm import tqdm
import torch
import numpy as np
import cv2
from PIL import Image
from collections import defaultdict
from cerwsi.utils import is_bbox_inside,random_cut_square

def bbox_intersection_area(boxes1, boxes2):
    """
    计算 boxes1 中每个框与 boxes2 中每个框的交集面积。
    :param boxes1: Tensor of shape (M, 4)
    :param boxes2: Tensor of shape (N, 4)
    :return: Tensor of shape (M, N)
    """
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # 左上角
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  # 右下角
    wh = (rb - lt).clamp(min=0)  # 相交区域的宽高
    inter = wh[:, :, 0] * wh[:, :, 1]
    return inter

def imgName2patientId(df_data)->str:
    name2PID = defaultdict(str)
    for row in tqdm(df_data.itertuples(index=False), total=len(df_data), ncols=80):
        filename = os.path.basename(row.kfb_path)
        name2PID[filename] = row.patientId
    return name2PID

def box_enclosured(bbox, all_bboxes):
    '''bbox 是否被 all_bboxes 中的某一个包裹'''
    for parent_bbox in all_bboxes:
        if bbox[0] == parent_bbox[0] and bbox[1] == parent_bbox[1] and bbox[2] == parent_bbox[2] and bbox[3] == parent_bbox[3]:
            continue
        if is_bbox_inside(bbox, parent_bbox, 5):
            return True
    return False

def has_box_inside(bbox, all_bboxes):
    '''bbox 是否包裹 all_bboxes 中的某一个'''
    for child_bbox in all_bboxes:
        if is_bbox_inside(child_bbox, bbox, 5):
            return True
    return False

def process_noparent_ann(annitems, roi_type, sq_size = 500):
    backup_rois = []
    for annitem in annitems:
        x1,y1,x2,y2 = annitem['region']
        w,h = x2-x1, y2-y1
        rx1,ry1,rx2,ry2 = float('inf'),float('inf'),-float('inf'),-float('inf')
        for i in range(5):
            square_x1,square_y1 = random_cut_square((x1,y1,w,h), sq_size=sq_size)
            square_x2,square_y2 = square_x1+sq_size, square_y1+sq_size
            rx1,ry1 = min(rx1, square_x1),min(ry1, square_y1)
            rx2,ry2 = max(rx2, square_x2),max(ry2, square_y2)
        # 处理情况： annitem bbox 过于大，生成的 RoI 无法完全包裹它
        rw,rh = rx2-rx1, ry2-ry1
        if rw < w or rh < h:
            maxlen = int(max(w, h)+0.5)
            rx1,ry1 = random_cut_square((x1,y1,w,h), sq_size=maxlen)
            rx2,ry2 = rx1 + maxlen, ry1 + maxlen

        backup_rois.append(dict(
            sub_class=roi_type, region=[rx1,ry1,rx2,ry2],
            parent_id = -1
        ))
    merged_rois = merge_backup_rois(backup_rois)

    return merged_rois

def clip_roi_region(coords, max_xy, minlen = 1024):
    '''coords: [x1,y1,x2,y2]'''

    rx1,ry1,rx2,ry2 = coords
    rx1,ry1 = max(0, rx1), max(0, ry1)
    if max_xy is not None:
        rx2,ry2 = min(max_xy[0], rx2), min(max_xy[1], ry2)
    rw,rh = rx2-rx1, ry2-ry1
    if rw < minlen:
        rx1 = rx1 - (minlen-rw)
    if rh < minlen:
        ry1 = ry1 - (minlen-rh)
    return [rx1,ry1,rx2,ry2]


def merge_backup_rois(backup_rois, min_size=200):
    n = len(backup_rois)
    visited = [False] * n
    groups = []

    def merge_regions(regions):
        x1 = min(r[0] for r in regions)
        y1 = min(r[1] for r in regions)
        x2 = max(r[2] for r in regions)
        y2 = max(r[3] for r in regions)
        return [x1, y1, x2, y2]
    
    def get_intersection(region1, region2):
        x1 = max(region1[0], region2[0])
        y1 = max(region1[1], region2[1])
        x2 = min(region1[2], region2[2])
        y2 = min(region1[3], region2[3])
        if x2 > x1 and y2 > y1:
            return x1, y1, x2, y2
        return None

    def valid_merge(region1, region2, min_size=200):
        intersect = get_intersection(region1, region2)
        if intersect:
            w = intersect[2] - intersect[0]
            h = intersect[3] - intersect[1]
            return w >= min_size and h >= min_size
        return False

    def dfs(idx, group):
        visited[idx] = True
        group.append(idx)
        for j in range(n):
            if not visited[j] and valid_merge(backup_rois[idx]['region'], backup_rois[j]['region'], min_size):
                dfs(j, group)

    for i in range(n):
        if not visited[i]:
            group = []
            dfs(i, group)
            groups.append(group)

    merged_rois = []
    for group in groups:
        regions = [backup_rois[i]['region'] for i in group]
        merged_region = merge_regions(regions)
        merged_rois.append(dict(
            annid=int(str(uuid.uuid4().int)[:13]),
            sub_class=backup_rois[group[0]]['sub_class'],
            region=merged_region,
            parent_id=-1
        ))

    return merged_rois

def draw_roi_inWSI(roi_items, slide, save_path):
    swidth, sheight = slide.level_dimensions[-1]
    downsample_ratio = slide.level_downsamples[-1]
    
    WSI_map = np.zeros((sheight, swidth, 3), dtype=np.uint8)
    for item in roi_items:
        rbbox = item['region']  # x1,y1,x2,y2
        scaled_rbbox = (np.array(rbbox) / downsample_ratio).astype(int).tolist()
        x1, y1, x2, y2 = scaled_rbbox
        # 限制坐标在图像边界内
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(swidth - 1, x2), min(sheight - 1, y2)
        # 设置颜色：绿色 (0,255,0)，红色 (0,0,255)
        if item['sub_class'] == 'RoI':
            color = (0, 255, 0)
        elif item['sub_class'] == 'forge_RoI':
            color = (0, 0, 255)
        cv2.rectangle(WSI_map, (x1, y1), (x2, y2), color=color, thickness=-1)  # 实心矩形填充
    Image.fromarray(cv2.cvtColor(WSI_map, cv2.COLOR_BGR2RGB)).save(save_path)

def adjust_region4RoI(roi_region, roi_children):
    '''
    根据 roi_children 的坐标重新调整 roi_region 的坐标，
    确保 roi_region 能包裹住所有的 roi_children
    '''
    rx1,ry1,rx2,ry2 = roi_region
    for citem in roi_children:
        cx1,cy1,cx2,cy2 = citem['region']
        rx1,ry1 = min(rx1, cx1), min(ry1, cy1)
        rx2,ry2 = max(rx2, cx2), max(ry2, cy2)
    return rx1,ry1,rx2,ry2

def deduplicate_regions(rect_items, tolerance=5):
    """
    去重 bbox 列表，允许坐标有小误差（比如5px内）
    """
    kept_rects = []
    for rect in rect_items:
        is_duplicate = False
        for kept_rect in kept_rects:
            bbox = np.array(rect['region'])
            kept_bbox = np.array(kept_rect['region'])
            if np.all(np.abs(bbox - kept_bbox) <= tolerance):
                is_duplicate = True
                break
        if not is_duplicate:
            kept_rects.append(rect)
    return kept_rects
