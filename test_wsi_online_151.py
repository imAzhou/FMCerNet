import torchvision
torchvision.disable_beta_transforms_warning()
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=r"^xFormers is available \(.*\)")
import torch
import json
import time
from mmpretrain.structures import DataSample
import argparse
import cv2
import math
from torchvision.ops import nms
from mmengine.config import Config
from math import ceil
import numpy as np
from PIL import Image
import pandas as pd
import multiprocessing
from multiprocessing import Pool
import os
import copy
from torchvision import transforms
from mmengine.logging import MMLogger
from cerwsi.utils import KFBSlide, set_seed
from cerwsi.nets import ValidClsNet, PatchClsNet

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.conv")

LEVEL = 0
PATCH_EDGE = 512
CERTAIN_THR = 0.7
NEGATIVE_THR = 0.5
positive_ratio_thr = 0.05
POSITIVE_CLASS = ['AGC', 'ASC-US','LSIL', 'ASC-H', 'HSIL']

def inference_valid_batch(valid_model, read_result_pool, curent_id, save_prefix):
    data_batch = dict(inputs=[], data_samples=[])
    for read_result in read_result_pool:
        img_input = cv2.cvtColor(np.array(read_result), cv2.COLOR_RGB2BGR)
        img_input = torch.as_tensor(cv2.resize(img_input, (224,224)))
        data_batch['inputs'].append(img_input.permute(2,0,1))    # (bs, 3, h, w)
        data_batch['data_samples'].append(DataSample())

    data_batch['inputs'] = torch.stack(data_batch['inputs'], dim=0)
    with torch.no_grad():
        outputs = valid_model.val_step(data_batch)
    
    valid_idx = []
    for idx,pred_output in enumerate(outputs):
        o_img = read_result_pool[idx]
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
        
        if args.visual_pred and n_tag in args.visual_pred:
            new_save_prefix = f'{save_prefix}/{n_tag}'
            os.makedirs(new_save_prefix, exist_ok=True)
            o_img.save(f'{new_save_prefix}_{sum(curent_id)}.png')

    return valid_idx

def format_bboxes(outputs, bidx, downsample_ratio, pcoords):
    sx1,sy1,sx2,sy2 = pcoords   # patch 块在 Slide 中的坐标
    token_probs,token_classes = outputs['token_probs'][bidx],outputs['token_classes'][bidx]
    feat_size = int(math.sqrt(token_probs.shape[0]))
    token_classes = token_classes.reshape((feat_size, feat_size)).detach().cpu().numpy()
    token_classes_resized = cv2.resize(token_classes, (PATCH_EDGE, PATCH_EDGE), interpolation=cv2.INTER_NEAREST)
    token_probs = token_probs.reshape((feat_size, feat_size)).detach().cpu().numpy()
    token_probs_resized = cv2.resize(token_probs, (PATCH_EDGE, PATCH_EDGE), interpolation=cv2.INTER_LINEAR)
    total_bboxes = []
    for class_id in range(1, len(POSITIVE_CLASS) + 1):  # 只处理 > 0 的类别
        mask = (token_classes_resized == class_id)
        if np.sum(mask) > 0:
            cls_name = POSITIVE_CLASS[class_id - 1]
            # 获取连通区域的外接矩形框
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
            bboxes = []
            scores = []
            for i in range(1, num_labels):  # 跳过背景
                x, y, bwidth, bheight, _ = stats[i]
                x1, y1, x2, y2 = x, y, x + bwidth, y + bheight
                if bwidth < 50 and bheight < 50:
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    x1, y1 = center_x-25,center_y-25
                    x2, y2 = center_x+25,center_y+25
                
                # 提取第 i 个连通区域的 mask（按 label）
                region_mask = (labels == i)
                region_probs = token_probs_resized[region_mask]
                max_score = float(np.max(region_probs)) if region_probs.size > 0 else 0.0
                bboxes.append((np.array([x1, y1, x2, y2]))*downsample_ratio)   # bbox 坐标 (在 LEVEL=0 上的坐标)
                scores.append(max_score)

            boxes_tensor = torch.tensor(np.array(bboxes), dtype=torch.float32)
            scores_tensor = torch.tensor(scores, dtype=torch.float32)
            keep = nms(boxes_tensor, scores_tensor, iou_threshold=0.7)
            kept_boxes = boxes_tensor[keep].numpy()
            kept_boxes_score = scores_tensor[keep].numpy()
            
            for coord,score in zip(kept_boxes, kept_boxes_score):
                bx1,by1,bx2,by2 = coord.tolist()
                total_bboxes.append({
                    'coord': [bx1+sx1,by1+sy1,bx2+sx1,by2+sy1],
                    'score': float(score),
                    'clsname': cls_name,
                })
    class_order = POSITIVE_CLASS[::-1]  # ['HSIL', 'ASC-H', 'LSIL', 'ASC-US', 'AGC']
    # 排序：先按 clsname 在 class_order 中的索引，再按 score 降序
    total_bboxes_sorted = sorted(
        total_bboxes,
        key=lambda x: (class_order.index(x['clsname']), -x['score'])
    )
    if len(total_bboxes) > 20:
        top_bbox = total_bboxes_sorted[0]
        
        total_bboxes_sorted = [{
            'coord': pcoords,
            'score': top_bbox['score'],
            'clsname': top_bbox['clsname'],
        }]   
    return total_bboxes_sorted

def inference_batch_pn(pn_model, valid_input, save_prefix, downsample_ratio):
    transform = transforms.Compose([
        transforms.Resize(pn_model.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    img_inputs = [transform(item['image']) for item in valid_input]
    images_tensor = torch.stack(img_inputs, dim=0).to(pn_model.device)
    data_batch = dict(
        inputs=images_tensor,
        data_samples = [DataSample(ori_shape=(PATCH_EDGE,PATCH_EDGE)) for i in range(len(valid_input))]
    )

    with torch.no_grad():
        outputs = pn_model(data_batch, 'val')
    
    pred_result = []
    for bidx in range(len(outputs['inputs'])):
        pcoords = valid_input[bidx]['coords']
        # pred_cls = torch.max(outputs['token_classes'][bidx], dim=-1)[0]
        # pred_clsid = (pred_cls > 0).int().item()
        pred_clsid = (outputs['img_probs'][bidx] > 0.5).int().item()

        patch_predinfo = {
            'pred_label': pred_clsid,
            'pos_bboxes': []
        }
        # if pred_clsid == 1:
        #     patch_predinfo['pos_bboxes'] = format_bboxes(outputs, bidx, downsample_ratio, pcoords)
        pred_result.append(patch_predinfo)
        
        if args.visual_pred and str(pred_clsid) in args.visual_pred:
            o_img = valid_input[bidx]['image']
            timestamp = time.time()
            os.makedirs(f'{save_prefix}/{pred_clsid}', exist_ok=True)
            o_img.save(f'{save_prefix}/{pred_clsid}/{timestamp}.png')
    return pred_result

def process_patches(proc_id, start_points, valid_model, pn_model, kfb_path, patientId):   
    save_prefix = f'{args.pred_save_dir}/{patientId}'
    
    slide = KFBSlide(kfb_path)
    downsample_ratio = slide.level_downsamples[LEVEL]
    read_result_pool, valid_read_result = [], []
    curent_id = [0,0,0]
    total_pred_results = []
    
    for p_idx,(x,y)in enumerate(start_points):
        location, level, size = (x, y), LEVEL, (PATCH_EDGE, PATCH_EDGE)
        read_result = copy.deepcopy(Image.fromarray(slide.read_region(location, level, size)))
        coords = np.array([x, y, x+PATCH_EDGE, y+PATCH_EDGE])*downsample_ratio
        read_result_pool.append({
            'image': read_result,
            'coords': coords.tolist(),   # patch 坐标 (在 LEVEL=0 上的坐标)
            'pn_pred': [],  # 该patch块的阳性bbox预测项（仅valid patch有此值）
        })
        
        if len(read_result_pool) % args.test_bs == 0 or p_idx == len(start_points)-1:
            imgs = [item['image'] for item in read_result_pool]
            valid_idx = inference_valid_batch(valid_model, imgs, curent_id, save_prefix)
            valid_read_result.extend([read_result_pool[idx] for idx in valid_idx])
            read_result_pool = []
            print(f'\rCore: {proc_id}, 当前已处理: {sum(curent_id)}', end='')
        
        if len(valid_read_result) > 0 and not args.only_valid:
            pred_result = inference_batch_pn(pn_model, valid_read_result, save_prefix, downsample_ratio)
            for pidx, pitem in enumerate(valid_read_result):
                pitem['pn_pred'] = pred_result[pidx]
                del pitem['image']
            total_pred_results.extend(valid_read_result)
            valid_read_result = []

    print(f'Core: {proc_id}, process {sum(curent_id)} patches done!!')
    return curent_id, total_pred_results

def get_pn_model(device):
    cfg = Config.fromfile(args.config_file)
    cfg.backbone_cfg['backbone_ckpt'] = None
    cfg.instance_ckpt = None
    model = PatchClsNet(cfg).to(device)
    model.img_size = cfg.input_size
    model.load_ckpt(args.ckpt)
    model.eval()
    return model

def multiprocess_inference():
    all_kfb_info = pd.read_csv(args.test_csv_file)

    device = torch.device(args.device)
    valid_model = ValidClsNet()
    valid_model.to(device)
    valid_model.eval()
    valid_model.load_state_dict(torch.load(args.valid_model_ckpt))

    if args.only_valid:
        pn_model = None
    else:
        pn_model = get_pn_model(device)

    print('='*10 + 'Models Load Done!' + '='*10)
    if args.record_save_dir:
        os.makedirs(args.record_save_dir, exist_ok=True)
    logger = MMLogger.get_instance('test_wsi', log_file=f'{args.record_save_dir}/test_wsi_v21.log')

    low_valid_kfb_info = []
    for row in all_kfb_info.itertuples(index=True):
        # if row.Index+1 <= 189:
        #     continue
        start_time = time.time()
        print('collecting start points... ')
        slide = KFBSlide(row.kfb_path)
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
        
        cpu_num = args.cpu_num
        set_split = np.array_split(slide_start_points, cpu_num)
        print(f"Number of cores: {cpu_num}, set number of per core: {len(set_split[0])}")
        workers = Pool(processes=cpu_num)
        # pool_state = Manager().dict()
        # pool_state["valid_cnt"] = [0,0,0]
        processes = []
        for proc_id, set_group in enumerate(set_split):
            p = workers.apply_async(process_patches,
                                    (proc_id, set_group, valid_model, pn_model,
                                     row.kfb_path, row.patientId))
            processes.append(p)

        valid_result, pn_result = [], []
        for p in processes:
            valids,pns = p.get()
            valid_result.append(valids)
            pn_result.extend(pns)
        workers.close()
        workers.join()

        t_delta = time.time() - start_time
        curent_id = np.array(valid_result)

        if args.only_valid:
            logger.info(f'\n[{row.Index+1}/{len(all_kfb_info)}] Time of {row.patientId}: {t_delta:0.2f}s, invalid: {np.sum(curent_id[:,0])}, uncertain: {np.sum(curent_id[:,2])}, valid: {np.sum(curent_id[:,1])}, total: {len(slide_start_points)}')
        else:
            patch_pred = [res['pn_pred'] for res in pn_result]
            pn_result_patch_clsid = [predinfo['pred_label'] for predinfo in patch_pred]

            # 打印每张 slide 阴阳 patch 预测结果
            p_path_num = int(sum(pn_result_patch_clsid))
            n_patch_num = len(pn_result_patch_clsid) - p_path_num
            p_ratio = p_path_num / (p_path_num + n_patch_num + 1e-6)    # 防止除0
            pred_clsid = int(p_ratio > positive_ratio_thr)
            logger.info(f'\n[{row.Index+1}/{len(all_kfb_info)}] Time of {row.patientId}: {t_delta:0.2f}s, invalid: {np.sum(curent_id[:,0])}, uncertain: {np.sum(curent_id[:,2])}, valid: {np.sum(curent_id[:,1])}(positive:{p_path_num} negative:{n_patch_num} p_ratio:{p_ratio:0.4f} pred/gt:{pred_clsid}/{row.kfb_clsid}-{row.kfb_clsname})')

            # 保存每张patch预测的所有阳性bbox信息
            # pred_pos_items_save_dir = f'{args.record_save_dir}/pred_pos_items'
            # os.makedirs(pred_pos_items_save_dir, exist_ok=True)
            # with open(f'{pred_pos_items_save_dir}/{row.patientId}.json', 'w') as f:
            #     json.dump(pn_result, f)

        if np.sum(curent_id[:,1]) <= 200 or np.sum(curent_id[:,0]) > np.sum(curent_id[:,1])*2:
            low_valid_kfb_info.append([row.kfb_path, row.kfb_clsid, row.kfb_clsname, row.patientId, row.kfb_source])

    df_low_valid = pd.DataFrame(low_valid_kfb_info, columns=['kfb_path', 'kfb_clsid', 'kfb_clsname', 'patientId', 'kfb_source'])
    df_low_valid.to_csv(f'{args.record_save_dir}/low_valid.csv', index=False)

parser = argparse.ArgumentParser()
parser.add_argument('test_csv_file', type=str)
parser.add_argument('valid_model_ckpt', type=str)
parser.add_argument('config_file', type=str)
parser.add_argument('ckpt', type=str)
parser.add_argument('--record_save_dir', type=str)
parser.add_argument('--only_valid', action='store_true')
parser.add_argument('--visual_pred', type=str, nargs='*', choices=['0', '1', 'invalid', 'valid', 'uncertain'])
parser.add_argument('--pred_save_dir', type=str)
parser.add_argument('--cpu_num', type=int, default=1, help='multiprocess cpu num')
parser.add_argument('--test_bs', type=int, default=16, help='batch size of model test')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--device', type=str, default='cuda:0')
args = parser.parse_args()

if __name__ == '__main__':
    set_seed(args.seed)
    multiprocessing.set_start_method('spawn', force=True)
    multiprocess_inference()

'''
Time of process kfb elapsed: 805.35 seconds, valid: 6126, invalid: 1108, uncertain: 72, total: 7306
Time of process kfb elapsed: 71.05 seconds, valid: 6126, invalid: 1108,  uncertain: 72, total: 7306

CUDA_VISIBLE_DEVICES=1 python test_wsi_online_151.py \
    data_resource/0511/4_pure_train_negslide_v2.csv \
    checkpoints/valid_cls_best.pth \
    log/WINDOW_SIZE_512/mAP_30.6/config.py \
    log/WINDOW_SIZE_512/mAP_30.6/checkpoints/best.pth \
    --record_save_dir log/WINDOW_SIZE_512/mAP_30.6 \
    --cpu_num 8 \
    --test_bs 8 \
    --visual_pred 1 \
    --pred_save_dir log/WINDOW_SIZE_512/mAP_30.6/posInNeg \
    --only_valid \
    --visual_pred invalid valid 1

CUDA_VISIBLE_DEVICES=2 python test_wsi_online.py \
    data_resource/0416/annofiles/val.csv \
    checkpoints/valid_cls_best.pth \
    log/l_cerscan_v3/wscer_partial/config.py \
    log/l_cerscan_v3/wscer_partial/checkpoints/best.pth \
    --record_save_dir log/patientImgs_val \
    --cpu_num 8 \
    --test_bs 16 \
    --visual_pred valid \
    --pred_save_dir data_resource/0416/patientImgs/val \
    --only_valid
'''