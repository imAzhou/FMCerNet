'''
从每张 Slide 中截取 1000 张 patch，截取优先级如下：
1. 医生人工标注的 patch
2. PNClsNet 预测 阳性概率 > 0.5 的patch
3. PNClsNet 预测 阳性概率 < 0.5 的patch
4. ValidClsNet 预测 0.3 < 有效概率 < 0.7 的patch
5. ValidClsNet 预测 有效概率 < 0.3 的patch
'''
import copy
import os
import json
import glob
import pandas as pd
from tqdm import tqdm
from cerwsi.utils import KFBSlide,set_seed
from PIL import Image
from math import ceil
import numpy as np
import torch
from mmpretrain.structures import DataSample
from cerwsi.nets import ValidClsNet, PNClsNet
import multiprocessing
from multiprocessing import Pool
import cv2

PATCH_EDGE = 500
CERTAIN_THR = 0.7
POSITIVE_THR = 0.5
CUTNUM = 1000
NEGATIVE_CLASS = ['NILM', 'GEC']
ASC_CLASS = ['ASC-US', 'LSIL', 'ASC-H', 'HSIL', 'SCC']
AGC_CLASS = ['AGC-NOS', 'AGC', 'AGC-N', 'AGC-FN']
cut_save_dir = 'data_resource/slide_cls'
cpu_num = 1
test_bs = 32

def cut_anno_patch():
    with open('data_resource/cls_pn/1117_anno_train.json','r') as f:
        train_data = json.load(f)
    with open('data_resource/cls_pn/1117_anno_val.json','r') as f:
        val_data = json.load(f)

    for anno_data in [train_data, val_data]:
        for slide_valid_item in tqdm(anno_data['valid_imgs'], ncols=80):
            patientId = slide_valid_item['patientId']
            slide = KFBSlide(slide_valid_item['kfb_path'])

            for idx, img_item in enumerate(slide_valid_item['valid_anno']):
                patch_clsname = img_item['patch_clsname']
                x1,y1,x2,y2 = img_item['coord']
                location, level, size = (x1,y1), 0, img_item['size']
                read_result = Image.fromarray(slide.read_region(location, level, size))
                filename = f'{patientId}_anno{idx}.png'
                _save_dir = f'{cut_save_dir}/{patientId}/{patch_clsname}'
                os.makedirs(_save_dir, exist_ok=True)
                read_result.save(f'{_save_dir}/{filename}')

def load_models():
    set_seed(1234)
    valid_model_ckpt = 'checkpoints/vlaid_cls_best.pth'
    pn_model_ckpt = 'checkpoints/pn_cls_best/wxl1_resnet50.pth'
    device = torch.device('cuda:0')
    valid_model = ValidClsNet()
    valid_model.to(device)
    valid_model.eval()
    valid_model.load_state_dict(torch.load(valid_model_ckpt))
    
    pn_model = PNClsNet()
    pn_model.to(device)
    pn_model.eval()
    pn_state_dict = torch.load(pn_model_ckpt)
    if 'state_dict' in pn_state_dict:
        pn_state_dict = pn_state_dict['state_dict']
    pn_model.load_state_dict(pn_state_dict)
    print('='*10 + 'Models Load Done!' + '='*10)
    return valid_model,pn_model

def inference_valid_batch(valid_model, read_result_pool):
    data_batch = dict(inputs=[], data_samples=[])
    for start_point, read_result in read_result_pool:
        img_input = cv2.cvtColor(np.array(read_result), cv2.COLOR_RGB2BGR)
        img_input = torch.as_tensor(cv2.resize(img_input, (224,224)))
        data_batch['inputs'].append(img_input.permute(2,0,1))    # (bs, 3, h, w)
        data_batch['data_samples'].append(DataSample())

    data_batch['inputs'] = torch.stack(data_batch['inputs'], dim=0)
    with torch.no_grad():
        outputs = valid_model.val_step(data_batch)
    
    _valids, _invalids, _uncertains = [],[],[]
    for idx,pred_output in enumerate(outputs):
        if max(pred_output.pred_score) > CERTAIN_THR:
            if pred_output.pred_label == 1:
                _valids.append(read_result_pool[idx])
            else:
                _invalids.append(read_result_pool[idx])
        else:
            _uncertains.append(read_result_pool[idx])

    return _valids, _invalids, _uncertains

def inference_batch_pn(pn_model, valid_input):
    data_batch = dict(inputs=[], data_samples=[])
    for start_point, read_result in valid_input:
        img_input = cv2.cvtColor(np.array(read_result), cv2.COLOR_RGB2BGR)
        img_input = torch.as_tensor(cv2.resize(img_input, (224,224)))
        data_batch['inputs'].append(img_input.permute(2,0,1))    # (bs, 3, h, w)
        data_batch['data_samples'].append(DataSample())
    
    data_batch['inputs'] = torch.stack(data_batch['inputs'], dim=0)
    with torch.no_grad():
        outputs = pn_model.val_step(data_batch)
    pred_pos,pred_neg = [],[]
    for idx,pred_output in enumerate(outputs):
        pred_clsid = int(pred_output.pred_score[1] > POSITIVE_THR)
        if pred_clsid == 1:
            pred_pos.append(valid_input[idx])
        else:
            pred_neg.append(valid_input[idx])

    return pred_pos,pred_neg

def process_patches(proc_id, start_points, valid_model, pn_model,kfb_path):
    slide = KFBSlide(kfb_path)
    read_result_pool, valid_read_result = [], []
    pred_p_points,pred_n_points,pred_vunc_points,pred_vn_points = [],[],[],[]
    curent_process = 0
    for (x,y) in start_points:
        location, level, size = (x, y), 0, (PATCH_EDGE, PATCH_EDGE)
        read_result = copy.deepcopy(Image.fromarray(slide.read_region(location, level, size)))
        read_result_pool.append(((x,y), read_result))
        curent_process += 1

        if len(read_result_pool) % test_bs == 0:
            valids, invalids, uncertains = inference_valid_batch(valid_model, read_result_pool)
            valid_read_result.extend(valids)
            
            pred_vunc_points.extend([item[0] for item in uncertains])
            pred_vn_points.extend([item[0] for item in invalids])
            read_result_pool = []
            print(f'\rCore: {proc_id}, 当前已处理: {curent_process}', end='')
        
        if len(valid_read_result) > 0:
            positives,negatives = inference_batch_pn(pn_model, valid_read_result)
            pred_p_points.extend([item[0] for item in positives])
            pred_n_points.extend([item[0] for item in negatives])
            valid_read_result = []
    
    if len(read_result_pool) > 0:
        valids, invalids, uncertains = inference_valid_batch(valid_model, read_result_pool)
        pred_vunc_points.extend([item[0] for item in uncertains])
        pred_vn_points.extend([item[0] for item in invalids])
        if len(valids) > 0:
            positives,negatives = inference_batch_pn(pn_model, valids)
            pred_p_points.extend([item[0] for item in positives])
            pred_n_points.extend([item[0] for item in negatives])

    del read_result_pool
    torch.cuda.empty_cache()
    print(f'Core: {proc_id}, process {curent_process} patches done!!')
    return pred_p_points, pred_n_points, pred_vunc_points, pred_vn_points


def PNClsNet_cut(rowInfo, anno_nums):
    slide = KFBSlide(rowInfo.kfb_path)
    width, height = slide.level_dimensions[0]
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
    set_split = np.array_split(slide_start_points, cpu_num)
    print(f"Number of cores: {cpu_num}, set number of per core: {len(set_split[0])}")
    workers = Pool(processes=cpu_num)
    valid_model,pn_model = load_models()
    processes = []
    for proc_id, set_group in enumerate(set_split):
        p = workers.apply_async(process_patches,
                                (proc_id, set_group, valid_model, pn_model,rowInfo.kfb_path))
        processes.append(p)

    pred_results = [[],[],[],[]]    # p,n,valid_unc,valid_n
    for p in processes:
        results = p.get()
        for idx in range(len(pred_results)):
            pred_results[idx].extend(results[idx])
    workers.close()
    workers.join()

    return_results = [0]*len(pred_results)
    current_nums = anno_nums
    for sps, tag, idx in zip(pred_results, ['pred_p', 'pred_n', 'pred_vunc', 'pred_vn'],range(len(pred_results))):
        break_flag = False
        for start_point in sps:
            location, level, size = start_point, 0, (PATCH_EDGE, PATCH_EDGE)
            read_result = Image.fromarray(slide.read_region(location, level, size))
            _save_dir = f'{cut_save_dir}/{rowInfo.patientId}/{tag}'
            os.makedirs(_save_dir,exist_ok=True)
            filename = f'{rowInfo.patientId}_pred{current_nums}.png'
            read_result.save(f'{_save_dir}/{filename}')
            current_nums += 1
            return_results[idx] += 1
            if current_nums == CUTNUM:
                break_flag = True
                break
        
        if break_flag:
            break
    return return_results

def cut_from_net():
    df_train = pd.read_csv('data_resource/cls_pn/1117_train.csv')
    df_val = pd.read_csv('data_resource/cls_pn/1117_val.csv')

    for df_data, mode in zip([df_train, df_val], ['train', 'val']):
        mode_records = []
        for row in tqdm(df_data.itertuples(index=False), total=len(df_data)):
            patientId = row.patientId
            anno_nums = len(glob.glob(f'{cut_save_dir}/{patientId}/**/*.png'))
            num_results = [0]*4
            if anno_nums < CUTNUM:
                num_results = PNClsNet_cut(row, anno_nums)
            row_list = list(row)
            row_list.extend([anno_nums]+num_results)
            mode_records.append(row_list)
        
        new_columns = ['anno', 'pred_p', 'pred_n', 'pred_valid_unc', 'pred_valid_n']
        df_records = pd.DataFrame(mode_records, columns=df_data.columns.tolist() + new_columns)
        df_records.to_csv(f'{cut_save_dir}/{mode}.csv', index=False)

if __name__ == '__main__':
    
    # cut_anno_patch()
    multiprocessing.set_start_method('spawn', force=True)
    cut_from_net()
