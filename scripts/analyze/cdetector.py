import json
import cv2
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
# from cerwsi.utils import draw_OD,generate_cut_regions,is_bbox_inside

POSITIVE_CLASS = ['ASC-US', 'LSIL', 'ASC-H', 'HSIL', 'AGC']
colors = plt.cm.tab10(np.linspace(0, 1, len(POSITIVE_CLASS)))[:, :3] * 255
category_colors = {cat: tuple(map(int, color)) for cat, color in zip(POSITIVE_CLASS, colors)}
WINDOW_SIZE = 750
STRIDE = WINDOW_SIZE - 50

def get_cutregion_inside(square_coord, inside_items, min_overlap=50):
    '''
    Args:
        square_coord: patch块在原图中的坐标(x1, y1, x2, y2)
        inside_items: 原图中所有的标注框信息
        min_overlap: 标注框与patch块的最小交集阈值，若大于该阈值，则此标注框属于该patch块
    Return:
        item_inside_filter: 在patch块中或与patch块有交集区域的标注框信息，且标注框坐标已经更新为相对于patch块左上角的相对坐标
    '''
    item_inside_filter = []
    for idx,i in enumerate(inside_items):
        region = i.get('region')
        w,h = region['width'],region['height']
        x1,y1 = region['x'],region['y']
        x2,y2 = x1+w, y1+h
        if w<5 or h<5:
            continue
        sx1,sy1,sx2,sy2 = square_coord
        
        if is_bbox_inside([x1,y1,x2,y2], square_coord, tolerance=5):
            relative_bbox = [x1 - sx1, y1 - sy1, x2 - sx1, y2 - sy1]
            rx1,ry1,rx2,ry2 = relative_bbox
            relative_region = dict(x=rx1, y=ry1, width=rx2-rx1, height=ry2-ry1)
            item_inside_filter.append(dict(sub_class=i['sub_class'], region=relative_region))
            continue
        
        # 计算交集区域
        inter_x_min = max(x1, sx1)
        inter_y_min = max(y1, sy1)
        inter_x_max = min(x2, sx2)
        inter_y_max = min(y2, sy2)

        if inter_x_min < inter_x_max and inter_y_min < inter_y_max:
            inter_w = inter_x_max - inter_x_min
            inter_h = inter_y_max - inter_y_min
            if inter_w > min_overlap and inter_h > min_overlap:
                relative_bbox = [inter_x_min - sx1, inter_y_min - sy1, inter_x_max - sx1, inter_y_max - sy1]
                rx1,ry1,rx2,ry2 = relative_bbox
                relative_region = dict(x=rx1, y=ry1, width=rx2-rx1, height=ry2-ry1)
                item_inside_filter.append(dict(sub_class=i['sub_class'], region=relative_region))
    
    return item_inside_filter

def convert_anno(annoinfo):
    clsname_map = {
        'ascus': 'ASC-US',
        'lsil': 'LSIL',
        'asch': 'ASC-H',
        'hsil': 'HSIL',
        'scc': 'HSIL',
        'agc': 'AGC',
        'trichomonas': 'NILM',
        'candida': 'NILM',
        'flora': 'NILM',
        'herps': 'NILM',
        'actinomyces': 'NILM',
    }
    categories_names = [item['name'] for item in annoinfo['categories']]
    anno_img = {}

    for item in annoinfo['annotations']:
        x,y,w,h = item['bbox']
        sub_class = clsname_map[categories_names[item['category_id']-1]]
        annoitem = {
            'id': item['id'],
            'sub_class': sub_class, 
            'region':{'x':x,'y':y,'width':w,'height':h}
        }
        if item['image_id'] in anno_img.keys():
            if sub_class in POSITIVE_CLASS:
                anno_img[item['image_id']].append(annoitem)
        else:
            anno_img[item['image_id']] = []
            if sub_class in POSITIVE_CLASS:
                anno_img[item['image_id']].append(annoitem)
    
    return anno_img
    
def analyze_img_wh():
    for mode in ['train','test']:
        with open(f'{root_dir}/{mode}.json', 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        all_w = [i['width'] for i in json_data['images']]
        all_h = [i['height'] for i in json_data['images']]
        
        print(f'{mode}: ')
        print(f'min w: {min(all_w)}, min h: {min(all_h)}')
        print(f'max w: {max(all_w)}, max h: {max(all_h)}')
        '''
        train: 
        min w: 975, min h: 580
        max w: 2100, max h: 1253
        test: 
        min w: 975, min h: 580
        max w: 2100, max h: 1253
        '''

def analayze_img_nums():
    for mode in ['train', 'test']:
        image_dir = f'{root_dir}/{mode}'
        json_path = f'{root_dir}/{mode}.json'
        with open(json_path, 'r') as f:
            annoinfo = json.load(f)
        anno_img = convert_anno(annoinfo)
        
        pos_nums,neg_nums = 0,0
        for fidx, filename in enumerate(tqdm(os.listdir(image_dir))):
            img = cv2.imread(f'{image_dir}/{filename}')
            h,w,_ = img.shape
            inside_items = anno_img[filename]

            cut_points = generate_cut_regions((0,0), w, h, WINDOW_SIZE, STRIDE)
            for iidx,rect_coords in enumerate(cut_points):
                x1,y1 = rect_coords
                x2,y2 = x1+WINDOW_SIZE,y1+WINDOW_SIZE
                item_inside_patch = get_cutregion_inside([x1,y1,x2,y2], inside_items)
                if len(item_inside_patch) == 0:
                    neg_nums += 1
                else:
                    pos_nums += 1
        print(f'{mode}: window size: {WINDOW_SIZE}, pos_nums: {pos_nums}, neg_nums: {neg_nums}')
        '''
        train: window size: 500, pos_nums: 32488, neg_nums: 16180
        test: window size: 500, pos_nums: 3687, neg_nums: 1949
        
        train: window size: 406, pos_nums: 36690, neg_nums: 28776, total: 65466
        test: window size: 406, pos_nums: 4117, neg_nums: 3571, total: 7688
        total: pos_nums: 40807, neg_nums: 32347
        '''

def analyze_cls_dist():
    for mode in ['train', 'val']:
        json_path = f'{root_dir}/annofiles/{mode}_patches.json'
        with open(json_path, 'r') as f:
            annofile = json.load(f)
        cls_cnt = [0]*len(POSITIVE_CLASS)
        for imganno in tqdm(annofile, ncols=80):
            pos_clsid = list(set([i[-1] for i in imganno['gtmap_14']]))
            for clsid in pos_clsid:
                cls_cnt[clsid-1] += 1
        print(f'{mode}: {cls_cnt}')

    '''
    含某类别的总图像样本数量，例如：训练集中有3666张图片内含类别ASC-US
    clsname: ['ASC-US', 'LSIL', 'ASC-H', 'HSIL', 'AGC']
    train: [3666, 3481, 5044, 21979, 6454]
    val: [413, 393, 557, 2289, 869]
    '''


if __name__ == '__main__':

    root_dir = 'data_resource/ComparisonDetectorDataset'
    analyze_img_wh()
    # vis_sample_img()
    # analayze_img_nums()
    # analyze_cls_dist()