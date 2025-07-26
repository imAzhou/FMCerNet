import torchvision
torchvision.disable_beta_transforms_warning()
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=r"^xFormers is available \(.*\)")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.conv")
import time
import argparse
import numpy as np
import pandas as pd
import multiprocessing
from multiprocessing import Pool
import os
from mmengine.logging import MMLogger
from cerwsi.utils import KFBSlide, set_seed
from cerwsi.utils.job_utils import (inference_valid_batch,inference_batch_pn,get_models,
                                 collect_startpoints,pred_postprocess,read_patch_fn,LEVEL)


def process_patches(proc_id, start_points, valid_model, pn_model, kfb_path, patientId):
    save_prefix = f'{args.pred_save_dir}/{patientId}'
    
    slide = KFBSlide(kfb_path)
    downsample_ratio = slide.level_downsamples[LEVEL]
    read_result_pool, valid_read_result = [], []
    curent_id = [0,0,0]
    total_pred_results = []
    
    for p_idx,point_xy in enumerate(start_points):
        read_patchItem = read_patch_fn(slide, point_xy, downsample_ratio)
        read_result_pool.append(read_patchItem)
        
        if len(read_result_pool) % args.test_bs == 0 or p_idx == len(start_points)-1:
            valid_idx,curent_id = inference_valid_batch(valid_model, read_result_pool, curent_id, args.visual_pred, save_prefix)
            valid_read_result.extend([read_result_pool[idx] for idx in valid_idx])
            read_result_pool = []
            print(f'\rCore: {proc_id}, 当前已处理: {sum(curent_id)}', end='')
        
        if len(valid_read_result) > 0 and not args.only_valid:
            pred_result = inference_batch_pn(pn_model, valid_read_result, downsample_ratio, args.visual_pred, save_prefix)
            for pidx, pitem in enumerate(valid_read_result):
                pitem['pn_pred'] = pred_result[pidx]
                del pitem['image']
            total_pred_results.extend(valid_read_result)
            valid_read_result = []

    print(f'Core: {proc_id}, process {sum(curent_id)} patches done!!')
    return curent_id, total_pred_results


def multiprocess_inference():
    all_kfb_info = pd.read_csv(args.test_csv_file)
    pn_model, valid_model = get_models()
    print('='*10 + 'Models Load Done!' + '='*10)
    if args.record_save_dir:
        os.makedirs(args.record_save_dir, exist_ok=True)
    logger = MMLogger.get_instance('test_wsi', log_file=f'{args.record_save_dir}/test_wsi.log')

    low_valid_kfb_info = []
    for row in all_kfb_info.itertuples(index=True):
        # if row.Index+1 <= 181:
        #     continue
        start_time = time.time()
        slide_start_points = collect_startpoints(row.kfb_path)
        
        cpu_num = args.cpu_num
        set_split = np.array_split(slide_start_points, cpu_num)
        print(f"Number of cores: {cpu_num}, set number of per core: {len(set_split[0])}")
        workers = Pool(processes=cpu_num)
        # pool_state = Manager().dict()
        # pool_state["valid_cnt"] = [0,0,0]
        processes = []
        for proc_id, set_group in enumerate(set_split):
            p = workers.apply_async(process_patches,(proc_id, set_group, valid_model, pn_model,row.kfb_path, row.patientId))
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
            slide_predInfo = pred_postprocess(pn_result)
            pred_clsid = 0 if slide_predInfo['pred_pos_items'] == 'subCls' else 1
            p_num,n_num,p_ratio = slide_predInfo['p_patch_num'],slide_predInfo['n_patch_num'],slide_predInfo['p_ratio']

            logger.info(f'\n[{row.Index+1}/{len(all_kfb_info)}] Time of {row.patientId}: {t_delta:0.2f}s, invalid: {np.sum(curent_id[:,0])}, uncertain: {np.sum(curent_id[:,2])}, valid: {np.sum(curent_id[:,1])}(positive:{p_num} negative:{n_num} p_ratio:{p_ratio:0.4f} pred/gt:{pred_clsid}/{row.kfb_clsid}-{row.kfb_clsname})')

        if np.sum(curent_id[:,1]) <= 200 or np.sum(curent_id[:,0]) > np.sum(curent_id[:,1])*2:
            low_valid_kfb_info.append([row.kfb_path, row.kfb_clsid, row.kfb_clsname, row.patientId, row.kfb_source])

    df_low_valid = pd.DataFrame(low_valid_kfb_info, columns=['kfb_path', 'kfb_clsid', 'kfb_clsname', 'patientId', 'kfb_source'])
    df_low_valid.to_csv(f'{args.record_save_dir}/low_valid.csv', index=False)

parser = argparse.ArgumentParser()
parser.add_argument('test_csv_file', type=str)
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

CUDA_VISIBLE_DEVICES=1 python tools/test_wsi_online.py \
    data_resource/0630/6_val.csv \
    --record_save_dir log/WS1600 \
    --cpu_num 4 \
    --test_bs 4 \
    --visual_pred 1 \
    --pred_save_dir log/WS1600/posInNeg \
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