import json
import cv2
import os
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from tqdm import tqdm
from cerwsi.utils import generate_cut_regions,is_bbox_inside

POSITIVE_CLASS = ['ASC-US', 'LSIL', 'ASC-H', 'HSIL', 'AGC']
classes = ['negative', *POSITIVE_CLASS]
WINDOW_SIZE = 406   # 要保证 WINDOW_SIZE 是 featmap_size 的整数倍
STRIDE = 350
featmap_size = 14

def find_matching_bboxes(target_bbox, grid_size, stride, min_overlap=50):
    matching_bboxes = []
    x1_min, y1_min, x1_max, y1_max = target_bbox

    # 计算col 和 row 范围
    col_min = int(max(0, x1_min // stride))
    col_max = int(x1_max // stride)
    row_min = int(max(0, y1_min // stride))
    row_max = int(y1_max // stride)
    
    for row in range(row_min, row_max + 1):
        for col in range(col_min, col_max + 1):
            x2_min = col * stride
            y2_min = row * stride
            x2_max = x2_min + grid_size
            y2_max = y2_min + grid_size
            bbox = [x2_min, y2_min, x2_max, y2_max]
            # 判断完全包含 (target_bbox完全在该patch内部)
            if is_bbox_inside(target_bbox, bbox):
                relative_bbox = [x1_min - x2_min, y1_min - y2_min, x1_max - x2_min, y1_max - y2_min]
                matching_bboxes.append(([row,col], relative_bbox))
                continue

            # 计算交集区域
            inter_x_min = max(x1_min, x2_min)
            inter_y_min = max(y1_min, y2_min)
            inter_x_max = min(x1_max, x2_max)
            inter_y_max = min(y1_max, y2_max)

            if inter_x_min < inter_x_max and inter_y_min < inter_y_max:
                inter_w = inter_x_max - inter_x_min
                inter_h = inter_y_max - inter_y_min
                if inter_w > min_overlap and inter_h > min_overlap:
                    relative_bbox = [inter_x_min - x2_min, inter_y_min - y2_min, inter_x_max - x2_min, inter_y_max - y2_min]
                    matching_bboxes.append(([row,col], relative_bbox))

    return matching_bboxes

def draw_tokenGT(read_result, img_savepath, gtmap, inside_items):
    # 创建绘图区域
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # 处理子图一：绘制图像和网格
    box_scale = 224 / read_result.size[0]
    img_resized = read_result.resize((224, 224))  # resize 到 224x224
    sq_x1,sq_y1,sq_w,sq_h = 0,0,224,224
    
    draw = ImageDraw.Draw(img_resized)
    grid_size = 16
    for i in range(0, 224, grid_size):
        draw.line([(i, 0), (i, 224)], fill="black", width=1)  # 竖线
        draw.line([(0, i), (224, i)], fill="black", width=1)  # 横线
    for box_item in inside_items:
        category = box_item.get('sub_class')
        region = box_item.get('region')
        x,y = region['x'],region['y']
        w,h = region['width'],region['height']
        x1, y1, x2, y2 = x,y,x+w,y+h
        x1, y1, x2, y2 = x1*box_scale, y1*box_scale, x2*box_scale, y2*box_scale
        x_min = max(sq_x1, x1) - sq_x1
        y_min = max(sq_y1, y1) - sq_y1
        x_max = min(sq_x1+sq_w, x2) - sq_x1
        y_max = min(sq_y1+sq_h, y2) - sq_y1
        
        color = (205, 92, 92)
        draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=2)
        draw.text((x_min + 2, y_min - 15), category, fill=color)
    axes[0].imshow(img_resized)
    axes[0].axis('off')  # 隐藏坐标轴
    
    # 处理子图二：类别 ID 映射成色块显示
    gtmap_array = np.array([i[-1] for i in gtmap])
    cmap = plt.get_cmap('tab20')  # 使用 Matplotlib 的 20 类别颜色
    unique_classes = np.unique(gtmap_array)
    color_map = {cls_id: cmap(idx / len(unique_classes)) for idx, cls_id in enumerate(unique_classes)}
    
    # 创建一个新的图像，每个类别使用不同颜色
    color_gtmap = np.zeros((featmap_size, featmap_size, 3), dtype=np.uint8)
    for row,col,clsid in gtmap:
        color = color_map[clsid]
        color_rgb = (np.array(color[:3]) * 255).astype(np.uint8)
        color_gtmap[row,col,:] = color_rgb
    
    color_gtmap_img = Image.fromarray(color_gtmap).resize((224, 224), resample=Image.Resampling.NEAREST)  # resize 到 224x224
    draw_gtmap = ImageDraw.Draw(color_gtmap_img)
    for i in range(0, 224, grid_size):
        draw_gtmap.line([(i, 0), (i, 224)], fill="white", width=1)  # 竖线
        draw_gtmap.line([(0, i), (224, i)], fill="white", width=1)  # 横线
    axes[1].imshow(color_gtmap_img)
    axes[1].axis('off')  # 隐藏坐标轴

    # 保存并关闭绘图
    plt.tight_layout()
    plt.savefig(img_savepath)
    plt.close()

def get_cutregion_inside(square_coord, inside_items, min_overlap=50):
    '''
    Args:
        square_coord: patch块在原图中的坐标(x1, y1, x2, y2)
        inside_items: 原图中所有的标注框信息
        min_overlap: 标注框与patch块的最小交集阈值，若大于该阈值，则此标注框属于该patch块
    Return:
        item_inside_filter: 在patch块中或与patch块有交集区域的标注框信息，且标注框坐标已经更新为相对于patch块左上角的相对坐标, eg: [dict(sub_class='ASC-US', region=dict(x=0, y=0, width=66, height=77))]
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
    
def makeGT(bboxes,clsnames):
    gtmap = []
    grid_size = WINDOW_SIZE // featmap_size
    stride = grid_size
    # 计算面积和索引，并按面积从大到小排序
    areas_with_indices = sorted(
        enumerate(bboxes),
        key=lambda x: -((x[1][2] - x[1][0]) * (x[1][3] - x[1][1]))  # 加负号实现降序排序
    )
    sorted_indices = [idx for idx, _ in areas_with_indices]
    sorted_bbox_list = [bbox for _, bbox in areas_with_indices]
    sorted_clsnames = [clsnames[idx] for idx in sorted_indices]

    for idx in range(len(bboxes)):
        x1,y1,x2,y2 = sorted_bbox_list[idx]
        x1,y1,x2,y2 = round(x1),round(y1),round(x2),round(y2)
        clsname = sorted_clsnames[idx]
        clsid = classes.index(clsname)
        
        matched_grids = find_matching_bboxes(sorted_bbox_list[idx], grid_size, stride, min_overlap=10)
        for (row,col),_ in matched_grids:
            gtmap.append([row, col, clsid])
    return gtmap
    
def gene_img_json():
    pos_save_dir = f'{root_dir}/images/Pos'
    os.makedirs(pos_save_dir, exist_ok=True)
    neg_save_dir = f'{root_dir}/images/Neg'
    os.makedirs(neg_save_dir, exist_ok=True)
    annojson_save_dir = f'{root_dir}/annofiles'
    os.makedirs(annojson_save_dir, exist_ok=True)

    for mode in ['train', 'test']:
    # for mode in ['train']:
        image_dir = f'{root_dir}/{mode}'
        json_path = f'{root_dir}/{mode}.json'
        with open(json_path, 'r') as f:
            annoinfo = json.load(f)
        anno_img = convert_anno(annoinfo)

        patch_list = []
        for fidx, filename in enumerate(tqdm(os.listdir(image_dir))):
            # if fidx > 10:
            #     break
            img = cv2.imread(f'{image_dir}/{filename}')
            h,w,_ = img.shape
            img_np = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            inside_items = anno_img[filename]
            purename = filename.split('.')[0]

            cut_points = generate_cut_regions((0,0), w, h, WINDOW_SIZE, STRIDE)
            for iidx,rect_coords in enumerate(cut_points):
                x1,y1 = rect_coords
                x2,y2 = x1+WINDOW_SIZE,y1+WINDOW_SIZE
                item_inside_patch = get_cutregion_inside([x1,y1,x2,y2], inside_items)
                filename = f'{purename}_{iidx}.png'
                annoitem = {'filename': filename, 'gtmap_14': []}
                if len(item_inside_patch) == 0:
                    save_path = f'{neg_save_dir}/{filename}'
                    annoitem['diagnose'] = 0
                    annoitem['prefix'] = 'Neg'
                else:
                    save_path = f'{pos_save_dir}/{filename}'
                    annoitem['diagnose'] = 1
                    annoitem['prefix'] = 'Pos'
                    bboxes = [[
                        i['region']['x'], i['region']['y'],
                        i['region']['x'] + i['region']['width'],
                        i['region']['y'] + i['region']['height'],
                        ] for i in item_inside_patch]
                    clsnames = [i['sub_class'] for i in item_inside_patch]
                    annoitem['gtmap_14'] = makeGT(bboxes,clsnames)

                img_patch = Image.fromarray(img_np[y1:y2, x1:x2, :])
                img_patch.save(save_path)

                # sample_save_dir = f'statistic_results/cdetector/{purename}'
                # os.makedirs(sample_save_dir, exist_ok=True)
                # draw_tokenGT(img_patch, f'{sample_save_dir}/tokenGT_{filename}', annoitem['gtmap_14'], item_inside_patch)

                patch_list.append(annoitem)
        
        with open(f'{annojson_save_dir}/{mode}_patches.json', 'w') as f:
            json.dump(patch_list, f)
        

if __name__ == '__main__':
    # root_dir = '/x22201018/datasets/CervicalDatasets/ComparisonDetectorDataset'
    # root_dir = '/c22073/zly/datasets/CervicalDatasets/ComparisonDetectorDataset'
    root_dir = '/disk/medical_datasets/cervix/ComparisonDetectorDataset'
    
    gene_img_json()
