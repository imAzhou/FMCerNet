import os
import glob
from PIL import Image
from cerwsi.utils import KFBSlide, random_cut_fn
import xml.etree.ElementTree as ET
from tqdm import tqdm
import numpy as np
import pandas as pd
import random
import torch
from cerwsi.nets import ValidClsNet
import cv2
from mmpretrain.structures import DataSample


def decode_xml(xml_path,max_x, max_y):
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
        x2,y2 = min(x2,max_x), min(y2,max_y)

        w, h = x2 - x1, y2 - y1
        if w > 32 and h > 32:
            cut_results = random_cut_fn(x1,y1,w,h)
            for new_rect in cut_results:
                new_x1,new_y1,new_w,new_h = new_rect
                result = dict(
                    location = (new_x1, new_y1),
                    size = (new_w, new_h),
                    original_box = (x1,y1,w,h)
                )
                all_rects.append(result)
    return all_rects


def patch_level_with_anno():
    img_save_dir = 'data_resource/cls_pn'
    train_csv = pd.read_csv(f'{img_save_dir}/train.csv')
    val_csv = pd.read_csv(f'{img_save_dir}/val.csv')
    filtered = {
        'train': train_csv[train_csv['kfb_clsid'] == 1],
        'val': val_csv[val_csv['kfb_clsid'] == 1],
    }

    for mode in ['train','val']:
        mode_txt_lines = []
        for kfb_path in filtered[mode]['kfb_path']:
            xml_path = kfb_path.replace('.kfb','.xml')
            clsname = kfb_path.split('/')[-2]
            patientId = 'wxl1_' + os.path.basename(kfb_path).split('.')[0]
            
            rc_save_path = f'{img_save_dir}/random_cut/{clsname}'
            os.makedirs(rc_save_path, exist_ok=True)
            o_save_path = f'{img_save_dir}/original/{clsname}'
            os.makedirs(o_save_path, exist_ok=True)

            slide = KFBSlide(kfb_path)
            max_x, max_y = slide.level_dimensions[0]
            all_rects = decode_xml(xml_path, max_x, max_y)

            for j,rect in enumerate(tqdm(all_rects, ncols=80)):
                location, level, size = rect['location'], 0, rect['size']
                read_result = Image.fromarray(slide.read_region(location, level, size))
                filename = f'{patientId}_{j}.png'
                read_result.save(f'{rc_save_path}/{filename}')

                mode_txt_lines.append(f'{clsname}/{filename} 1\n')

                o_x1,o_y1,o_w,o_h = rect['original_box']
                original_read_result = Image.fromarray(slide.read_region((o_x1,o_y1), 0, (o_w,o_h)))
                original_read_result.save(f'{o_save_path}/{filename}')
        with open(f'{img_save_dir}/random_cut/pos_{mode}.txt', 'w') as txtf:
            txtf.writelines(mode_txt_lines)


def patch_level_with_NILM():
    device = torch.device('cuda:0')
    valid_model_ckpt = 'checkpoints/vlaid_cls_best.pth'
    valid_model = ValidClsNet()
    valid_model.to(device)
    valid_model.eval()
    valid_model.load_state_dict(torch.load(valid_model_ckpt))
    CERTAIN_THR = 0.7

    root_dir = 'data_resource/cls_pn'
    img_save_dir = f'{root_dir}/random_cut/NILM'
    os.makedirs(img_save_dir, exist_ok=True)
    train_csv = pd.read_csv(f'{root_dir}/train.csv')
    val_csv = pd.read_csv(f'{root_dir}/train.csv')
    filtered = {
        'train': train_csv[train_csv['kfb_clsid'] == 0],
        'val': val_csv[val_csv['kfb_clsid'] == 0],
    }

    for mode in ['train','val']:
        mode_txt_lines = []
        valid_idx = 0
        for kfb_path in filtered[mode]['kfb_path']:
            patientId = 'wxl1_' + os.path.basename(kfb_path).split('.')[0]
            slide = KFBSlide(kfb_path)
            max_x, max_y = slide.level_dimensions[0]
            # 100 张宽高在[100,300]， 200 张宽高在[300,600]
            small_cut_num, large_cut_num = 100, 200
            if mode == 'val':
                small_cut_num, large_cut_num = 50, 100
            for random_cut_num in [(small_cut_num, [100,300]), (large_cut_num, [300,600])]:
                for i in tqdm(range(random_cut_num[0]), ncols=80):
                    x1,y1 = random.randint(0, max_x),random.randint(0, max_y)
                    w = random.randint(random_cut_num[1][0], random_cut_num[1][1])
                    h = int(w*random.uniform(0.8, 1.5))
                    if x1+w > max_x or y1+h > max_y:
                        continue
                    read_result = Image.fromarray(slide.read_region((x1,y1), 0, (w,h)))

                    img_input = cv2.cvtColor(np.array(read_result), cv2.COLOR_RGB2BGR)
                    img_input = torch.as_tensor(cv2.resize(img_input, (224,224)))
                    data_batch = dict(inputs=[], data_samples=[])
                    data_batch['inputs'].append(img_input.permute(2,0,1))    # (bs, 3, h, w)
                    data_batch['data_samples'].append(DataSample())
                    data_batch['inputs'] = torch.stack(data_batch['inputs'], dim=0)
                    with torch.no_grad():
                        outputs = valid_model.val_step(data_batch)
                    if max(outputs[0].pred_score) > CERTAIN_THR and outputs[0].pred_label == 1:
                        filename = f'{patientId}_{valid_idx}.png'
                        read_result.save(f'{img_save_dir}/{filename}')
                        mode_txt_lines.append(f'NILM/{filename} 0\n')
                        valid_idx += 1
        
        with open(f'{root_dir}/random_cut/neg_{mode}.txt', 'w') as txtf:
            txtf.writelines(mode_txt_lines)


def split_kfbs():
    positive_csv = 'data_resource/cls_pn/anno_kfbs.csv'
    negative_csv = 'data_resource/cls_pn/val_NILM_kfbs.csv'
    train_ratio = 0.7
    svae_dir = 'data_resource/cls_pn'

    df_positive = pd.read_csv(positive_csv)
    df_negative = pd.read_csv(negative_csv)
    p_kfbs = df_positive['img_path'].tolist()
    n_kfbs = df_negative['img_path'].tolist()
    random.shuffle(p_kfbs)
    random.shuffle(n_kfbs)

    p_kfbs_train,p_kfbs_val = p_kfbs[:int(len(p_kfbs)*train_ratio)],p_kfbs[int(len(p_kfbs)*train_ratio):]
    n_kfbs_train,n_kfbs_val = n_kfbs[:int(len(n_kfbs)*train_ratio)],n_kfbs[int(len(n_kfbs)*train_ratio):]

    df_train_kfbs = pd.DataFrame([
        *[[kfb_path, 1] for kfb_path in p_kfbs_train],
        *[[kfb_path, 0] for kfb_path in n_kfbs_train],
    ], columns=['kfb_path', 'kfb_clsid'])
    df_train_kfbs.to_csv(f'{svae_dir}/train.csv', index=False)

    df_val_kfbs =pd.DataFrame([
        *[[kfb_path, 1] for kfb_path in p_kfbs_val],
        *[[kfb_path, 0] for kfb_path in n_kfbs_val],
    ], columns=['kfb_path', 'kfb_clsid'])
    df_val_kfbs.to_csv(f'{svae_dir}/val.csv', index=False)


def concat_pos_neg_txt():
    root_dir = 'data_resource/cls_pn/random_cut'
    for mode in ['train','val']:
        with open(f'{root_dir}/pos_{mode}.txt', 'r') as f:
            pos_lines = f.readlines()
        with open(f'{root_dir}/neg_{mode}.txt', 'r') as f:
            neg_lines = f.readlines()
        concat_lines = [*pos_lines, *neg_lines]
        random.shuffle(concat_lines)
        with open(f'{root_dir}/{mode}.txt', 'w') as txtf:
            txtf.writelines(concat_lines)


if __name__ == '__main__':
    # split_kfbs()
    # patch_level_with_anno()
    # patch_level_with_NILM()
    concat_pos_neg_txt()