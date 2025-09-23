from mmpretrain.structures import DataSample
from mmdet.structures import DetDataSample
import torch
import cv2
from PIL import Image
from torchvision import transforms
from math import ceil
import copy
from mmdet.models.detectors import DINO
from mmengine.config import Config
from mmengine.registry import init_default_scope
from cerwsi.nets import ValidClsNet
from cerwsi.utils import KFBSlide,is_bbox_inside
import numpy as np
from collections import Counter
import warnings
import os
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.conv")
LEVEL = 0
PATCH_EDGE = 1600
CERTAIN_THR = 0.7
NEGATIVE_THR = 0.5
BBOX_SCORE_THR = 0.2
POSITIVE_CLASS = ['AGC', 'ASC-US','LSIL', 'ASC-H', 'HSIL', 'SCC']
positive_ratio_thr = [0.05, 0.1]   # thr<0.05: 阴性， thr>0.1: 阳性， 0.1>thr>0.05: 疑似
device = "cuda:0"
root_dir = '/nfs5/zly/codes/python-backend/medmodels/CerWSI'
valid_model_ckpt = f"{root_dir}/checkpoints/valid_cls_best.pth"
config_file = f"{root_dir}/log/WS1600/vis_data/config.py"
pn_model_ckpt = f"{root_dir}/log/WS1600/epoch_8.pth"

def infer_valid_fn(valid_model, read_result_pool, curent_id, visual_pred=None, save_prefix=None):
    data_batch = dict(inputs=[], data_samples=[])
    imgs = [item['image'] for item in read_result_pool]
    for read_result in imgs:
        img_input = cv2.cvtColor(np.array(read_result), cv2.COLOR_RGB2BGR)
        img_input = torch.as_tensor(cv2.resize(img_input, (224,224)))
        data_batch['inputs'].append(img_input.permute(2,0,1))    # (bs, 3, h, w)
        data_batch['data_samples'].append(DataSample())

    data_batch['inputs'] = torch.stack(data_batch['inputs'], dim=0)
    with torch.no_grad():
        outputs = valid_model.val_step(data_batch)
        torch.cuda.synchronize()
    
    valid_idx = []
    for idx,pred_output in enumerate(outputs):
        if max(pred_output.pred_score) > CERTAIN_THR:
            if pred_output.pred_label == 1:
                n_tag = 'valid'
                curent_id[1] += 1
                valid_idx.append(idx)
            else:
                n_tag = 'invalid'
                curent_id[0] += 1
        else:
            n_tag = 'uncertain'
            curent_id[2] += 1
        
        if visual_pred and n_tag in visual_pred:
            o_img = read_result_pool[idx]
            new_save_prefix = f'{save_prefix}/{n_tag}'
            os.makedirs(new_save_prefix, exist_ok=True)
            o_img.save(f'{new_save_prefix}_{sum(curent_id)}.png')

    return valid_idx,curent_id


def infer_pn_fn(pn_model, valid_input, downsample_ratio, visual_pred=None, save_prefix=None):
    inputsize = pn_model.img_size
    imgw,imgh = PATCH_EDGE,PATCH_EDGE
    transform = transforms.Compose([
        transforms.Resize(inputsize),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    
    img_inputs = [transform(item['image']) for item in valid_input]
    images_tensor = torch.stack(img_inputs, dim=0).to(pn_model.device)
    data_batch = dict(
        inputs=images_tensor,
        data_samples = [
            DetDataSample(
                metainfo={
                    'img_shape':(inputsize,inputsize),
                    'ori_shape':(imgw,imgh),
                    'scale_factor': (inputsize/imgh, inputsize/imgh)
                },
                batch_input_shape=(inputsize,inputsize),
            ) for i in range(len(valid_input))]
    )

    with torch.no_grad():
        outputs = pn_model(data_batch['inputs'], data_batch['data_samples'], mode="predict")
    pred_result = []
    for bidx in range(len(img_inputs)):
        pcoords = valid_input[bidx]['coords']
        predresult = outputs[bidx].pred_instances
        format_pred = postprocess_pred(predresult, pcoords, downsample_ratio)
        pred_clsid = int(len(format_pred) > 0)
        patch_predinfo = {
            'pred_label': pred_clsid,
            'patch_coords': pcoords,
            'pos_bboxes': format_pred
        }
        pred_result.append(patch_predinfo)
        if visual_pred and str(pred_clsid) in visual_pred:
            o_img = valid_input[bidx]['image']
            timestamp = time.time()
            os.makedirs(f'{save_prefix}/{pred_clsid}', exist_ok=True)
            o_img.save(f'{save_prefix}/{pred_clsid}/{timestamp}.png')

    return pred_result

def get_models():
    init_default_scope('mmdet')
    cfg = Config.fromfile(config_file)
    del cfg.model.type
    pn_model = DINO(**cfg.model).to(device)
    pn_model.img_size = cfg.input_size
    pn_model.device = device
    ckpt = torch.load(pn_model_ckpt, weights_only=False, map_location=device)['state_dict']
    pn_model.load_state_dict(ckpt)
    pn_model.eval()

    init_default_scope('mmpretrain')
    valid_model = ValidClsNet()
    valid_model.to(device)
    valid_model.device = device
    valid_model.eval()
    valid_model.load_state_dict(torch.load(valid_model_ckpt))
    print("开始加载模型，路径为: " + valid_model_ckpt)

    return pn_model, valid_model

def postprocess_pred(predresult, pcoords, downsample_ratio):
    pred_bboxes,pred_scores,pred_labels = predresult.bboxes.cpu(), predresult.scores.cpu(), predresult.labels.cpu()
    px1, py1, px2, py2 = pcoords
    new_bboxes = []
    filtered_bboxes,filtered_scores,filtered_labels = [],[],[]
    for bbox,score,label in zip(pred_bboxes,pred_scores,pred_labels):
        if score >= BBOX_SCORE_THR:
            filtered_bboxes.append(bbox.tolist()) # x1,y1,x2,y2
            filtered_scores.append(score.item())
            filtered_labels.append(label.item())
    if len(filtered_bboxes) == 0:
        return new_bboxes
    
    bboxes,labels = np.array(filtered_bboxes),np.array(filtered_labels)
    areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    sorted_indices = np.argsort(-areas)  # 从大到小排序

    n = len(bboxes)
    used = np.zeros(n, dtype=bool)
    for i in sorted_indices:
        if used[i]:
            continue
        group_indices = [i]
        for j in sorted_indices:
            if i == j or used[j]:
                continue
            if is_bbox_inside(bboxes[j], bboxes[i], tolerance=5):
                group_indices.append(j)
                used[j] = True

        # 当前 group 最外层是 i，label 取 group 内最大值
        max_label = labels[group_indices].max()
        bx1, by1, bx2, by2 = (bboxes[i]*downsample_ratio).tolist()
        bx1, by1, bx2, by2 = bx1+px1, by1+py1, bx2+px1, by2+py1
        new_bboxes.append({
            'parent_coord': pcoords,   # bbox 所在 patch 块坐标（在 LEVEL=0 上的绝对坐标）
            'coord': [bx1, by1, bx2, by2],  # bbox 在 LEVEL=0 上的绝对坐标
            'score': filtered_scores[i],     # bbox 阳性类别置信度
            'clsname': POSITIVE_CLASS[int(max_label)]    # bbox 阳性类别名称
        })
        used[i] = True

    return new_bboxes

def collect_startpoints(kfb_path):
    print('collecting start points... ')
    slide = KFBSlide(kfb_path)
    width, height = slide.level_dimensions[LEVEL]
    iw, ih = ceil(width/PATCH_EDGE), ceil(height/PATCH_EDGE)
    r2 = (int(max(iw, ih)*1.1)//2)**2
    cix, ciy = iw // 2, ih // 2
    slide_start_points = []
    for j, y in enumerate(range(0, height, PATCH_EDGE)):
        for i, x in enumerate(range(0, width, PATCH_EDGE)):
            if (i-cix)**2 + (j-ciy)**2 > r2:
                continue
            slide_start_points.append((x, y))

    print(f'total start points: {len(slide_start_points)}')
    return slide_start_points

def pred_postprocess(patches_pred_result):
    category = '阴性'
    p_patch_num,n_patch_num,p_ratio = 0,0,0.
    pred_pos_items = [] # 存储 Slide 中预测的所有 阳性bbox，已经按照阳性程度 + 预测置信度排序
    if len(patches_pred_result) > 0:
        patch_pred = [res['pn_pred'] for res in patches_pred_result]
        pn_result_patch_clsid,pred_pos_items = [],[]
        for predinfo in patch_pred:
            pn_result_patch_clsid.append(predinfo['pred_label'])
            pred_pos_items.extend(predinfo['pos_bboxes'])

        p_patch_num = int(sum(pn_result_patch_clsid))
        n_patch_num = len(pn_result_patch_clsid) - p_patch_num
        p_ratio = p_patch_num / (p_patch_num + n_patch_num + 1e-6)    # 防止除0
        if p_ratio > positive_ratio_thr[0] and p_ratio < positive_ratio_thr[1]:
            category = '疑似'
        elif p_ratio > positive_ratio_thr[1]:
            category = '阳性'

    positive_class = [x['clsname'] for x in pred_pos_items]
    counter = Counter(positive_class)
    counter_str = ", ".join(f"{key}: {value}" for key, value in counter.items())
    subCls_list = list(set(positive_class))
    subCls = "阴性" if len(subCls_list) == 0 else ",".join(subCls_list)

    slide_predInfo = {
        'pred_pos_items':pred_pos_items,
        'category': category,
        'counter_str': counter_str,
        'subCls': subCls,
        'p_patch_num': p_patch_num,
        'n_patch_num': n_patch_num,
        'p_ratio': p_ratio,
    }

    return slide_predInfo

def read_patch_fn(slide, point_xy, downsample_ratio):
    x,y = point_xy
    location, level, size = (x, y), LEVEL, (PATCH_EDGE, PATCH_EDGE)
    read_result = copy.deepcopy(Image.fromarray(slide.read_region(location, level, size)))
    coords = np.array([x, y, x+PATCH_EDGE, y+PATCH_EDGE])*downsample_ratio

    return {
        'image': read_result,
        'coords': coords.tolist(),   # patch 坐标 (在 LEVEL=0 上的坐标)
        'pn_pred': [],  # 该patch块的阳性预测概率（仅valid patch有此值）
    }
