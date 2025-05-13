from tqdm import tqdm
import argparse
import numpy as np
import pandas as pd
import warnings
import os
import random
import json
import cv2
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from cerwsi.utils import (KFBSlide,remap_points,read_json_anno,is_bbox_inside,draw_OD)

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.conv")

PATCH_EDGE = 700
STRIDE = 650
featmap_size = 14
POSITIVE_CLASS = ['ASC-US', 'LSIL', 'ASC-H', 'HSIL', 'SCC', 'AGC-NOS', 'AGC', 'AGC-N', 'AGC-FN']
colors = plt.cm.tab10(np.linspace(0, 1, len(POSITIVE_CLASS)))[:, :3] * 255
category_colors = {cat: tuple(map(int, color)) for cat, color in zip(POSITIVE_CLASS, colors)}

classes = ['negative', 'ASC-US', 'LSIL', 'ASC-H', 'HSIL', 'AGC']
# 类别映射关系
RECORD_CLASS = {
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

def find_matching_bboxes(target_bbox, grid_size = PATCH_EDGE, stride = STRIDE, min_overlap=50):

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

def process_pos_slide(rowInfo):

    patches_result = {} # key is patch id, value is patch anno info

    pos_df = pd.read_csv('/nfs5/zly/codes/CerWSI/data_resource/ROI/annofile/1223_pos.csv')
    slide = KFBSlide(f'{args.data_root_dir}/{rowInfo.kfb_path}')
    swidth, sheight = slide.level_dimensions[0]
    total_cols = int(swidth // STRIDE) + 1
    # cut_points = generate_cut_regions((0,0), width, height, PATCH_EDGE, stride=50)
    patient_row = pos_df.loc[pos_df['patientId'] == rowInfo.patientId].iloc[0]
    json_path = f'{args.data_root_dir}/{patient_row.json_path}'
    annos = read_json_anno(json_path)
    for ann_ in annos:
        ann = remap_points(ann_)
        if ann is None:
            continue
        sub_class = ann.get('sub_class')
        region = ann.get('region')
        x,y = region['x'],region['y']
        w,h = region['width'],region['height']
        if w <=20 or h<=20 or sub_class not in POSITIVE_CLASS:
            continue
        if w>1000 or h>1000:
            continue
        target_bbox = [x,y,x+w,y+h]
        match_patches = find_matching_bboxes(target_bbox, min_overlap=50)
        for stp,coord in match_patches:
            row,col= stp
            patchid = row * total_cols + col
            
            sx1,sy1 = col * STRIDE, row * STRIDE
            sx2,sy2 = sx1+PATCH_EDGE, sy1+PATCH_EDGE
            if sx2 > swidth:
                exceed = sx2 - swidth
                sx1 = sx1 - exceed
                sx2 = swidth
                ix1,iy1,ix2,iy2 = coord
                coord = [ix1 - exceed,iy1,ix2 - exceed,iy2]
            if sy2 > sheight:
                exceed = sy2 - sheight
                sy1 = sy1 - exceed
                sy2 = sheight
                ix1,iy1,ix2,iy2 = coord
                coord = [ix1,iy1 - exceed,ix2,iy2 - exceed]

            if patchid not in patches_result.keys():
                patches_result[patchid] = {
                    'filename':f'{rowInfo.patientId}_{patchid}.png',
                    'square_x1y1': (sx1,sy1),
                    'bboxes': [coord],
                    'clsnames': [sub_class],
                    'diagnose': 1
                }
            else:
                patches_result[patchid]['bboxes'].append(coord)
                patches_result[patchid]['clsnames'].append(sub_class)
    
    return list(patches_result.values())

def makeGT(patch_list):
    result_list = []
    for patchinfo in patch_list:
        gtmap = np.zeros((featmap_size,featmap_size), dtype=int)   # (h,w)
        grid_size = PATCH_EDGE // featmap_size
        stride = grid_size

        # 计算面积和索引，并按面积从大到小排序
        areas_with_indices = sorted(
            enumerate(patchinfo['bboxes']),
            key=lambda x: -((x[1][2] - x[1][0]) * (x[1][3] - x[1][1]))  # 加负号实现降序排序
        )
        sorted_indices = [idx for idx, _ in areas_with_indices]
        sorted_bbox_list = [bbox for _, bbox in areas_with_indices]
        sorted_clsnames = [patchinfo['clsnames'][idx] for idx in sorted_indices]

        for idx in range(len(patchinfo['bboxes'])):
            x1,y1,x2,y2 = sorted_bbox_list[idx]
            x1,y1,x2,y2 = round(x1),round(y1),round(x2),round(y2)
            clsname = sorted_clsnames[idx]
            clsid = classes.index(RECORD_CLASS[clsname])
            
            matched_grids = find_matching_bboxes(sorted_bbox_list[idx], grid_size, stride, min_overlap=10)
            for (row,col),_ in matched_grids:
                gtmap[row, col] = clsid
        
        patchinfo['gtmap_14'] = gtmap.tolist()
        result_list.append(patchinfo)
    
    return result_list

def draw_tokenGT(read_result, img_savepath, gtmap):
    # 创建绘图区域
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # 处理子图一：绘制图像和网格
    img_resized = read_result.resize((224, 224))  # resize 到 224x224
    draw = ImageDraw.Draw(img_resized)
    grid_size = 16
    for i in range(0, 224, grid_size):
        draw.line([(i, 0), (i, 224)], fill="black", width=1)  # 竖线
        draw.line([(0, i), (224, i)], fill="black", width=1)  # 横线
    axes[0].imshow(img_resized)
    axes[0].axis('off')  # 隐藏坐标轴
    
    # 处理子图二：类别 ID 映射成色块显示
    gtmap_array = np.array(gtmap)
    cmap = plt.get_cmap('tab20')  # 使用 Matplotlib 的 20 类别颜色
    unique_classes = np.unique(gtmap_array)
    color_map = {cls_id: cmap(idx / len(unique_classes)) for idx, cls_id in enumerate(unique_classes)}
    
    # 创建一个新的图像，每个类别使用不同颜色
    color_gtmap = np.zeros((gtmap_array.shape[0], gtmap_array.shape[1], 3), dtype=np.uint8)
    for cls_id, color in color_map.items():
        color_rgb = (np.array(color[:3]) * 255).astype(np.uint8)
        color_gtmap[gtmap_array == cls_id] = color_rgb
    
    color_gtmap_img = Image.fromarray(color_gtmap).resize((224, 224), resample=Image.Resampling.NEAREST)  # resize 到 224x224
    draw_gtmap = ImageDraw.Draw(color_gtmap_img)
    for i in range(0, 224, grid_size):
        draw_gtmap.line([(i, 0), (i, 224)], fill="black", width=1)  # 竖线
        draw_gtmap.line([(0, i), (224, i)], fill="black", width=1)  # 横线
    axes[1].imshow(color_gtmap_img)
    axes[1].axis('off')  # 隐藏坐标轴

    # 保存并关闭绘图
    plt.tight_layout()
    plt.savefig(img_savepath)
    plt.close()


def gene_patch_json():
    os.makedirs(f'{args.save_dir}/images', exist_ok=True)
    os.makedirs(f'{args.save_dir}/annofiles', exist_ok=True)

    train_data_df = pd.read_csv(args.train_csv_file)
    val_data_df = pd.read_csv(args.val_csv_file)

    for data_df,mode in zip([train_data_df,val_data_df], ['train','val']):
        all_patch_list = []
        total_pos_nums = 0
        for row in tqdm(data_df.itertuples(index=True), total=len(data_df)):
            # if row.Index > 5:
            #     break
            if row.kfb_clsname != 'NILM':
                patch_list = process_pos_slide(row)
                total_pos_nums += len(patch_list)
                patch_list_withGT = makeGT(patch_list)
                
                all_patch_list.append({
                    'patientId': row.patientId,
                    'kfb_path': row.kfb_path,
                    'patch_list': patch_list_withGT
                })
        print(f'{mode}: {total_pos_nums} patches.')

        with open(f'{args.save_dir}/annofiles/{mode}_pos_patches.json', 'w') as f:
            json.dump(all_patch_list, f)

def cut_patch():
    for mode in ['train', 'val']:
        with open(f'{args.save_dir}/annofiles/{mode}_pos_patches.json', 'r') as f:
            kfb_list = json.load(f)
        
        for kfbinfo in tqdm(kfb_list, ncols=80):
            slide = KFBSlide(f'{args.data_root_dir}/{kfbinfo["kfb_path"]}')
            patch_list = kfbinfo['patch_list']
            
            for patchinfo in patch_list:
                x1,y1 = patchinfo['square_x1y1']
                location, level, size = (x1,y1), 0, (PATCH_EDGE,PATCH_EDGE)
                read_result = Image.fromarray(slide.read_region(location, level, size))
                read_result.save(f'{args.save_dir}/images/{patchinfo["filename"]}')

def vis_sample(vis_nums):
    sample_save_dir = 'statistic_results/0103/cut_pos_sample'
    os.makedirs(sample_save_dir, exist_ok=True)
    with open(f'{args.save_dir}/annofiles/train_pos_patches.json', 'r') as f:
        kfb_list = json.load(f)
    
    for idx, kfbinfo in enumerate(kfb_list):
        if idx == vis_nums:
            break
        slide = KFBSlide(f'{args.data_root_dir}/{kfbinfo["kfb_path"]}')
        patch_list = kfbinfo['patch_list']
        random.shuffle(patch_list)
        for pidx, patchinfo in enumerate(patch_list):
            if pidx == vis_nums:
                break
            x1,y1 = patchinfo['square_x1y1']
            innerbbox,bbox_clsname = patchinfo['bboxes'],patchinfo['clsnames']

            inside_items = []
            for coords,clsname in zip(innerbbox,bbox_clsname):
                cx1,cy1,cx2,cy2 = coords
                inside_items.append({
                    'sub_class': clsname,
                    'region': dict(x=cx1,y=cy1,width=cx2-cx1,height=cy2-cy1)
                })
            
            location, level, size = (x1,y1), 0, (PATCH_EDGE,PATCH_EDGE)
            read_result = Image.fromarray(slide.read_region(location, level, size))
            filename = patchinfo["filename"]
            square_coords = [0,0,PATCH_EDGE,PATCH_EDGE]
            draw_OD(read_result, f'{sample_save_dir}/{filename}', square_coords, inside_items,category_colors)
            draw_tokenGT(read_result, f'{sample_save_dir}/tokenGT_{filename}', patchinfo['gtmap_14'])

            

parser = argparse.ArgumentParser()
parser.add_argument('train_csv_file', type=str)
parser.add_argument('val_csv_file', type=str)
parser.add_argument('--valid_model_ckpt', type=str)
parser.add_argument('--data_root_dir', type=str, default='/medical-data/data')
parser.add_argument('--save_dir', type=str) # {save_dir}/images {save_dir}/annofiles

args = parser.parse_args()

if __name__ == '__main__':
    gene_patch_json()
    # vis_sample(vis_nums = 10)
    # cut_patch()

'''
python scripts/0103/slide_cut_pos.py \
    /nfs5/zly/codes/CerWSI/data_resource/ROI/annofile/1223_train.csv \
    /nfs5/zly/codes/CerWSI/data_resource/ROI/annofile/1223_val.csv \
    --save_dir /nfs5/zly/codes/CerWSI/data_resource/0103

    
train: 66158 patches.
val: 16202 patches.
'''