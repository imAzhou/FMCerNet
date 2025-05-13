import torch
import time
from mmpretrain.structures import DataSample
import argparse
import cv2
from math import ceil
import numpy as np
from PIL import Image
import pandas as pd
import multiprocessing
from multiprocessing import Pool
import warnings
import os
import copy
from mmengine.logging import MMLogger
from cerwsi.utils import (KFBSlide, set_seed,)
from cerwsi.nets import ValidClsNet, PatchClsNet, PatchClsDINO

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.conv")

PATCH_EDGE = 500
CERTAIN_THR = 0.7
NEGATIVE_THR = 0.7
positive_ratio_thr = 0.005
kfb_root_dir = '/medical-data/data'
POINT_FLAG = dict(valid=-1, invalid=0, uncertain=1, negative=2, positive=3)
CHOOSE_K = 1000

def inference_batch_valid(valid_model, read_result_pool):
    data_batch = dict(inputs=[], data_samples=[])
    for read_result in read_result_pool:
        img_input = cv2.cvtColor(np.array(read_result), cv2.COLOR_RGB2BGR)
        img_input = torch.as_tensor(cv2.resize(img_input, (224,224)))
        data_batch['inputs'].append(img_input.permute(2,0,1))    # (bs, 3, h, w)
        data_batch['data_samples'].append(DataSample())

    data_batch['inputs'] = torch.stack(data_batch['inputs'], dim=0)
    with torch.no_grad():
        outputs = valid_model.val_step(data_batch)
    
    valid_flag = []
    for idx,pred_output in enumerate(outputs):
        flag = POINT_FLAG['uncertain']
        if max(pred_output.pred_score) > CERTAIN_THR:
            flag = POINT_FLAG['valid'] if pred_output.pred_label == 1 else POINT_FLAG['invalid']
        valid_flag.append(flag)
    return valid_flag

def inference_batch_pn(pn_model, valid_input):
    data_batch = dict(inputs=[], data_samples=[])
    for read_result in valid_input:
        img_input = cv2.cvtColor(np.array(read_result), cv2.COLOR_RGB2BGR)
        img_input = torch.as_tensor(cv2.resize(img_input, (224,224)))
        data_batch['inputs'].append(img_input.permute(2,0,1))    # (bs, 3, h, w)
        data_batch['data_samples'].append(DataSample())
    
    data_batch['inputs'] = torch.stack(data_batch['inputs'], dim=0)
    with torch.no_grad():
        outputs = pn_model.val_step(data_batch)
    pn_flag = [
        POINT_FLAG['negative'] if int(pred_output.pred_score[0] > NEGATIVE_THR) else POINT_FLAG['positive'] 
        for pred_output in outputs]
    return pn_flag


def process_patches(proc_id, start_points, valid_model, pn_model, kfb_path):
    slide = KFBSlide(kfb_path)
    read_points_pool, read_result_pool, valid_read_result = [], [], [] 
    points_record = []  # list item: [valid_flag, x, y]
    
    for p_idx, (x,y) in enumerate(start_points):
        location, level, size = (x, y), 0, (PATCH_EDGE, PATCH_EDGE)
        read_result = copy.deepcopy(Image.fromarray(slide.read_region(location, level, size)))
        read_result_pool.append(read_result)
        read_points_pool.append((x,y))
        
        if len(read_result_pool) % args.test_bs == 0 or p_idx == len(start_points)-1:
            valid_flag = inference_batch_valid(valid_model, read_result_pool)
            points_record.extend([(f,x,y) for f, (x,y) in zip(valid_flag, read_points_pool)])
            valid_input = [read_result_pool[idx] for idx,flag in enumerate(valid_flag) if flag == POINT_FLAG['valid']]
            read_result_pool,read_points_pool = [],[]

            valid_read_result.extend(valid_input)
            print(f'\rCore: {proc_id}, 当前已处理: {p_idx+1}', end='')
        
        if len(valid_read_result) > 0:
            pn_flag = inference_batch_pn(pn_model, valid_read_result)
            pn_iter = iter(pn_flag)
            points_record = [[next(pn_iter), x[1], x[2]] if x[0] == -1 else x for x in points_record]
            valid_read_result = []

    del read_result_pool
    torch.cuda.empty_cache()
    print(f'Core: {proc_id}, process {p_idx+1} patches done!!')
    return points_record

def get_pn_model(device):
    if args.pn_model_type == 'resnet50':
        pn_model = PatchClsNet(num_classes = args.num_classes)
    elif args.pn_model_type == 'dinov2_s':
        pn_model = PatchClsDINO(num_classes = args.num_classes, device=device)
    
    pn_model.to(device)
    pn_model.eval()
    pn_state_dict = torch.load(args.pn_model_ckpt)
    if 'state_dict' in pn_state_dict:
        pn_state_dict = pn_state_dict['state_dict']
    pn_model.load_state_dict(pn_state_dict)

    return pn_model

def gene_token(proc_id, start_points, model, kfb_path, patientId):
    save_prefix = f'{args.token_save_dir}/pt/{patientId}'
    os.makedirs(save_prefix, exist_ok=True)
    
    slide = KFBSlide(kfb_path)
    pt_records = []
    for p_idx, (f,x,y) in enumerate(start_points):
        location, level, size = (x, y), 0, (PATCH_EDGE, PATCH_EDGE)
        read_result = Image.fromarray(slide.read_region(location, level, size))
        img_input = cv2.cvtColor(np.array(read_result), cv2.COLOR_RGB2BGR)
        img_input = torch.as_tensor(cv2.resize(img_input, (224,224))).permute(2,0,1)
        data_batch = dict(inputs=[img_input], data_samples=[DataSample()])
        data = model.data_preprocessor(data_batch, False)

        with torch.no_grad():
            outputs = model.extract_feat(data['inputs'])
        pt_filename = f'f{f}c{proc_id}p{p_idx}.pt'
        img_token = outputs[0].clone()
        torch.save(img_token, f'{save_prefix}/{pt_filename}')
        pt_records.append([pt_filename, x, y])
    return pt_records

def multiprocess_inference():
    all_kfb_info = pd.read_csv(args.test_csv_file)

    device = torch.device(args.device)
    valid_model = ValidClsNet()
    valid_model.to(device)
    valid_model.eval()
    valid_model.load_state_dict(torch.load(args.valid_model_ckpt))
    pn_model = get_pn_model(device)

    print('='*10 + 'Models Load Done!' + '='*10)
    os.makedirs(args.record_save_dir, exist_ok=True)
    os.makedirs(f'{args.token_save_dir}/pt_record', exist_ok=True)
    logger = MMLogger.get_instance('test_wsi', log_file=f'{args.record_save_dir}/test_wsi.log')

    for row in all_kfb_info.itertuples(index=True):
        start_time = time.time()
        print('collecting start points... ')
        slide = KFBSlide(f'{kfb_root_dir}/{row.kfb_path}')
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
        slide.close()
        print(f'total start points: {len(slide_start_points)}')
        
        cpu_num = args.cpu_num
        set_split = np.array_split(slide_start_points, cpu_num)
        print(f"Number of cores: {cpu_num}, set number of per core: {len(set_split[0])}")
        workers = Pool(processes=cpu_num)
        processes = []
        for proc_id, set_group in enumerate(set_split):
            p = workers.apply_async(process_patches,(proc_id, set_group, valid_model, pn_model, f'{kfb_root_dir}/{row.kfb_path}'))
            processes.append(p)
        points_record_total = []
        for p in processes:
            results = p.get()
            points_record_total.extend(results)
        workers.close()
        workers.join()

        np_prt = np.array(points_record_total)
        unique_f = np.unique(np_prt[:, 0])[::-1]
        shuffled_np_prt = []
        for f in unique_f:
            group = np_prt[np_prt[:, 0] == f]
            np.random.shuffle(group)
            shuffled_np_prt.append(group)
        sorted_np_prt = np.vstack(shuffled_np_prt)
        unique, counts = np.unique(sorted_np_prt[:, 0], return_counts=True)
        freq = dict(zip(unique, counts))
        
        choose_np_prt = sorted_np_prt[:CHOOSE_K]
        unique, counts = np.unique(choose_np_prt[:, 0], return_counts=True)
        chosen_freq = dict(zip(unique, counts))

        set_split = np.array_split(choose_np_prt, cpu_num)
        print(f"Number of cores: {cpu_num}, set number of per core: {len(set_split[0])}")
        print('Generating image token......')
        workers = Pool(processes=cpu_num)
        processes = []
        for proc_id, set_group in enumerate(set_split):
            p = workers.apply_async(gene_token,(proc_id, set_group, pn_model, f'{kfb_root_dir}/{row.kfb_path}', row.patientId))
            processes.append(p)
        pt_records = []
        for p in processes:
            results = p.get()
            pt_records.extend(results)
        # list(map(lambda f: f.get(), processes))
        workers.close()
        workers.join()

        df_pt_records = pd.DataFrame(pt_records, columns = ['pt_filename', 'start_x', 'start_y'])
        df_pt_records.to_csv(f'{args.token_save_dir}/pt_record/{row.patientId}.csv', index=False)

        t_delta = time.time() - start_time
        logger.info(f'\n[{row.Index+1}/{len(all_kfb_info)}] Time of {row.patientId}: {t_delta:0.2f}s, nums of flag: {freq}, chosen of flag: {chosen_freq}')

def test_token_right():
    device = torch.device(args.device)
    pn_model = get_pn_model(device)
    patientId = 'WXL_1_467'
    kfb_path = '/disk/medical_datasets/cervix/ZJU-TCT/第二批标注2023.9.5/NILM/01S246.kfb'
    slide = KFBSlide(kfb_path)
    x,y = 36000,7500
    saved_pt = torch.load('/disk/zly/slide_token/pt/WXL_1_467/f2c0p24.pt')
    location, level, size = (x, y), 0, (PATCH_EDGE, PATCH_EDGE)
    read_result = Image.fromarray(slide.read_region(location, level, size))
    img_input = cv2.cvtColor(np.array(read_result), cv2.COLOR_RGB2BGR)
    img_input = torch.as_tensor(cv2.resize(img_input, (224,224)))
    data_batch = dict(inputs=[img_input.permute(2,0,1)], data_samples=[DataSample()])
    
    data_batch['inputs'] = torch.stack(data_batch['inputs'], dim=0)
    with torch.no_grad():
        outputs = pn_model.val_step(data_batch)

parser = argparse.ArgumentParser()
parser.add_argument('test_csv_file', type=str)
parser.add_argument('valid_model_ckpt', type=str)
parser.add_argument('pn_model_type', type=str, choices=['resnet50', 'dinov2_s'])
parser.add_argument('pn_model_ckpt', type=str)
parser.add_argument('--record_save_dir', type=str)
parser.add_argument('--token_save_dir', type=str, default='/disk/zly/slide_token')
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--cpu_num', type=int, default=1, help='multiprocess cpu num')
parser.add_argument('--test_bs', type=int, default=16, help='batch size of model test')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--device', type=str, default='cuda:0')
args = parser.parse_args()

if __name__ == '__main__':
    set_seed(args.seed)
    multiprocessing.set_start_method('spawn', force=True)
    multiprocess_inference()

    # test_token_right()

'''

python scripts/split_data/gene_slide_token.py \
    data_resource/cls_pn/1127_val.csv \
    checkpoints/vlaid_cls_best.pth \
    resnet50 \
    checkpoints/rcp_c6_v2.pth \
    --record_save_dir log/1127_val_token \
    --num_classes 6 \
    --cpu_num 8 \
    --test_bs 64 \
    --token_save_dir data_resource/slide_token/val
'''