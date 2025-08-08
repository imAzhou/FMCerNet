import random
import numpy as np
import torch
import json
from PIL import ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import chardet
import xml.etree.ElementTree as ET

def set_seed(seed):
    # Set random seed for PyTorch
    torch.manual_seed(seed)

    # Set random seed for CUDA if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Set random seed for NumPy
    np.random.seed(seed)

    # Set random seed for random module
    random.seed(seed)

    # Set random seed for CuDNN if available
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    one_M = 1e6
    return {
        'Total': f'{(total_num/one_M):.4f}M',
        'Trainable': f'{(trainable_num/one_M):.4f}M',
    }

def is_bbox_inside(bbox1, bbox2, tolerance=0):
    """
    判断 bbox1 是否被包含在 bbox2 内部（允许一定误差）
    bbox 格式为 [x_min, y_min, x_max, y_max]
    
    参数:
        bbox1: list, 表示被检测的边界框
        bbox2: list, 表示容器边界框
        tolerance: float, 允许的超出误差值
    返回:
        bool: 如果 bbox1 在 bbox2 内（允许误差）则返回 True，否则返回 False
    """
    return (bbox1[0] >= bbox2[0] - tolerance and  # bbox1的左边界可以稍微超出bbox2的左边界
            bbox1[1] >= bbox2[1] - tolerance and  # bbox1的上边界可以稍微超出bbox2的上边界
            bbox1[2] <= bbox2[2] + tolerance and  # bbox1的右边界可以稍微超出bbox2的右边界
            bbox1[3] <= bbox2[3] + tolerance)     # bbox1的下边界可以稍微超出bbox2的下边界

def overlap_enough(bbox1, bbox2, min_overlap):
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    return inter_width > min_overlap and inter_height > min_overlap

def calc_relative_coord(parent_bbox, child_bbox, min_overlap=25):
    '''bbox: [x1,y1,x2,y2]'''
    relative_bbox = None
    px1,py1,px2,py2 = parent_bbox
    cx1,cy1,cx2,cy2 = child_bbox
    if is_bbox_inside(child_bbox, parent_bbox):
        relative_bbox = [cx1 - px1, cy1 - py1, cx2 - px1, cy2 - py1]
    else:
        # 计算交集区域
        inter_x_min = max(cx1, px1)
        inter_y_min = max(cy1, py1)
        inter_x_max = min(cx2, px2)
        inter_y_max = min(cy2, py2)

        if inter_x_min < inter_x_max and inter_y_min < inter_y_max:
            inter_w = inter_x_max - inter_x_min
            inter_h = inter_y_max - inter_y_min
            if inter_w > min_overlap and inter_h > min_overlap:
                relative_bbox = [inter_x_min - px1, inter_y_min - py1, inter_x_max - px1, inter_y_max - py1]
    return relative_bbox

def random_cut_fn(x1,y1,w,h, cut_num=1):
    if w < 64 or h < 64:
        maxlen = int(max(w,h) * random.uniform(1.5, 4))
        interval = sorted([128, maxlen])
    elif w > 224 and h > 224:
        maxlen = int(max(w,h) * random.uniform(1.1, 1.5))
        interval = [max(w,h), maxlen]
    else:
        maxlen = int(max(w,h) * random.uniform(1.5, 2))
        interval = [max(w,h), maxlen]
    
    cut_results = []
    for _ in range(cut_num):
        new_w,new_h = random.randint(interval[0],interval[1]),random.randint(interval[0],interval[1])
        minx,miny = x1-(new_w-w), y1-(new_h-h)
        maxx,maxy = x1, y1
        newx,newy = random.randint(minx,maxx), random.randint(miny,maxy)
        assert is_bbox_inside([x1,y1,x1+w,y1+h], [newx,newy,newx+new_w,newy+new_h]), "new box cannot contained the original box"
        cut_results.append([newx,newy,new_w,new_h])
    return cut_results

def random_cut_square(rect, sq_size):
    """
    根据输入矩形 rect (x, y, w, h) 的宽高条件，返回裁剪正方形区域的左上角坐标 (x1, y1)。
    sq_size：正方形边长
    规则：假设 sq_size = 500
    1. 如果 w > 500 且 h > 500，在矩形内部随机裁剪一块宽高为 500 的区域。
    2. 如果 w < 500 且 h < 500，随机生成一个宽高为 500 的矩形包裹住输入矩形。
    3. 如果宽或高小于 500，则生成一个宽高为 500 的矩形，包裹住短边，长边上随机。

    返回:
        tuple: 裁剪矩形区域的左上角坐标 (x1, y1)
    """
    int_rect = [round(i) for i in rect]
    x, y, w, h = int_rect
    # Case 1: Both width and height > sq_size
    if w > sq_size and h > sq_size:
        x1 = random.randint(x, x + w - sq_size)  # 随机选取左上角 x 坐标
        y1 = random.randint(y, y + h - sq_size)  # 随机选取左上角 y 坐标
        return x1, y1

    # Case 2: Both width and height < sq_size
    elif w < sq_size and h < sq_size:
        x1 = random.randint(x - sq_size + w, x)  # 左上角的 x1 随机
        y1 = random.randint(y - sq_size + h, y)  # 左上角的 y1 随机
        return x1, y1

    # Case 3: One side < sq_size
    else:
        min_len = min(w,h)
        if w == min_len:  # 宽度较短
            x1 = random.randint(x - sq_size + w, x)  # 左上角的 x1 随机
            y1 = random.randint(y, y + h - sq_size)  # 高度随机分布
        else:  # 高度较短
            x1 = random.randint(x, x + w - sq_size)  # 宽度随机分布
            y1 = random.randint(y - sq_size + h, y)  # 左上角的 y1 随机
        return x1, y1

def remap_points(annitem):
    points = annitem['points']
    if len(points) < 2:     # 判定为是从左上角从右下角绘制的矩形
        return annitem
    p1_x,p1_y = points[0]['x'],points[0]['y']
    p2_x,p2_y = points[1]['x'],points[1]['y']
    if p1_x < p2_x and p1_y < p2_y:
        return annitem
    if p1_x > p2_x and p1_y > p2_y:
        annitem['points'] = [points[1], points[0]]
        annitem['region'] = dict(
            x = p2_x, y = p2_y,
            width = p1_x - p2_x, height = p1_y - p2_y
        )
        return annitem
    
    return None

def decode_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    all_rects = []
    for region in root.findall('.//Region'):
        coords = []
        for vertex in region.findall('.//Vertex'):
            # 获取 Vertex 节点的 X 和 Y 属性值
            x = vertex.get('X')
            y = vertex.get('Y')
            coords.append((x,y))
        (start_x,start_y),(end_x, end_y) = coords[0],coords[2]
        start_x,start_y,end_x,end_y = round(float(start_x)), round(float(start_y)), round(float(end_x)), round(float(end_y))
        x1,y1 = min(start_x,end_x),min(start_y,end_y)
        x2,y2 = max(start_x,end_x),max(start_y,end_y)

        w, h = x2 - x1, y2 - y1
        if w > 32 and h > 32:
            all_rects.append([x1,y1,x2,y2])
    return all_rects

def read_json_anno(json_path, encoding='GB2312'):
    def detect_encoding(path):
        with open(path, 'rb') as f:
            sample = f.read(1024 * 1024)  # 读取前1MB数据进行检测
        result = chardet.detect(sample)
        detected_encoding = result['encoding']
        print(f"Detected Encoding: {detected_encoding}")
        return detected_encoding

    # 尝试解码 JSON 文件
    try:
        with open(json_path, 'r', encoding=encoding) as f:
            data = json.load(f)
    except (UnicodeDecodeError, TypeError):  # 解码错误时重新检测编码
        print(f"Failed to decode with encoding '{encoding}', trying auto-detect...")
        encoding = detect_encoding(json_path)
        with open(json_path, 'r', encoding=encoding) as f:
            data = json.load(f)

    annotations = data.get('annotation', [])
    return annotations

def generate_cut_regions(region_start, region_width, region_height, k, stride=400, minlen=0):
    """
    生成裁切区域的坐标框 [x1, y1, x2, y2]，按 stride 均匀划分。
    边缘不足 minlen 时舍弃

    :param region_start: 区域的起点坐标 (x, y)
    :param region_width: 区域的宽度
    :param region_height: 区域的高度
    :param k: 裁切区域边长（正方形）
    :param stride: 步长
    :param minlen: 边角区域的最小保留尺寸
    :return: 裁切区域坐标列表 [[x1, y1, x2, y2], ...]
    """
    x_start, y_start = region_start
    cut_regions = []

    # 1. 调整宽度
    w_rem = region_width % stride
    if w_rem <= minlen:
        region_width -= w_rem  # 舍弃不足 minlen 的部分
        new_width = region_width
    else:
        new_width = region_width + (stride-w_rem)
    # 2. 调整高度
    h_rem = region_height % stride
    if h_rem <= minlen:
        region_height -= h_rem
        new_height = region_height
    else:
        new_height = region_height + (stride-h_rem)

    # 3. 均匀取点
    for y in range(0, new_width, stride):
        for x in range(0, new_height, stride):
            x1, y1 = x, y
            x2, y2 = x1 + k, y1 + k

            # 4. 边界修正
            if x2 > region_width:
                x2 = region_width
                x1 = x2 - k
            if y2 > region_height:
                y2 = region_height
                y1 = y2 - k

            cut_regions.append([x1+x_start, y1+y_start, x2+x_start, y2+y_start])

    return cut_regions

def draw_OD(read_image, save_path, square_coords, inside_items, class_labels):
    '''
    square_coords: list|tuple, [x1,y1,w,h]
    inside_items: list[
        dict(sub_class:str,region:[ x1, y1, x2, y2])]
    class_labels: list([str])
    '''
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_labels)))[:, :3] * 255
    category_colors = {cat: tuple(map(int, color)) for cat, color in zip(class_labels, colors)}

    draw = ImageDraw.Draw(read_image)
    sq_x1,sq_y1,sq_w,sq_h = square_coords

    for box_item in inside_items:
        category = box_item.get('sub_class')
        x1, y1, x2, y2 = box_item.get('region')
        # x,y = region['x'],region['y']
        # w,h = region['width'],region['height']
        # x1, y1, x2, y2 = x,y,x+w,y+h
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
    # 获取图形的尺寸（单位：英寸）
    fig_width, fig_height = fig.get_size_inches()
    # 将图例偏移图像尺寸的1.5%
    offset_in_inches = 1.5 / fig_width

    ax.legend(handles=patches, loc='upper right', bbox_to_anchor=(1+offset_in_inches, 1), frameon=False)
    fig.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
