import json
import torch
import os
import shutil
import warnings
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
warnings.filterwarnings("ignore", category=UserWarning)
import torch.distributed as dist
import argparse
from mmengine.config import Config
from mmengine.registry import init_default_scope
from cerwsi.nets import PatchNet,ValidClsNet
from cerwsi.utils import set_seed, init_distributed_mode, is_main_process
from cerwsi.utils.wsi_handler import WSIHandler
from mmengine.logging import MMLogger
import pandas as pd
import random
import time

LEVEL,PATCH_EDGE = 0,850
CERTAIN_THR,POSITIVE_THR = 0.7,0.5
SEED,SAFE_MARGIN = 1234,100
test_bs,FP_savenum = 64,10
hs_round_idx = 2
valid_ckpt = 'checkpoints/valid_cls_best.pth'
FP_img_savedir = f'data_resource/0630/WINDOW_SIZE_{PATCH_EDGE}/images/neg_slide_r{hs_round_idx}'
FP_json_savepath = f'data_resource/0630/WINDOW_SIZE_{PATCH_EDGE}/ann_jsons/patches_in_negslide_hs{hs_round_idx}.json'
infer_csv_file = f'data_resource/0630/WINDOW_SIZE_{PATCH_EDGE}/annofiles/45_purejfsw_train.csv'

pnmodel_rootdir = 'log/WS850/hs_round1'
mmcls_config_file = f'{pnmodel_rootdir}/config.py'
mmcls_ckpt = f'{pnmodel_rootdir}/checkpoints/best.pth'
infer_log_savepath = f'{pnmodel_rootdir}/negslide_infer.log'
collect_result_savepath = f'{pnmodel_rootdir}/negslide_collect_result.txt'
tmp_save_dir = f'{pnmodel_rootdir}/tmp/mlc_hs_negslide'
os.makedirs(tmp_save_dir, exist_ok=True, mode=0o777)


def resume_infer():
    done_pidlist = []
    for filename in os.listdir(tmp_save_dir):
        pid = filename.split('.')[0]
        done_pidlist.append(pid)
    return done_pidlist

def run_inference(valid_model, mlcls_model):
    infer_results = []
    done_pidlist = resume_infer()
    df = pd.read_csv(infer_csv_file)
    df = df.drop_duplicates(subset=["patientId"])   # 按 patientId 去重
    df = df[df['kfb_clsid']==0]
    df = df[~df["patientId"].isin(done_pidlist)]
    data_list = df.to_dict(orient="records")  # 每一行 -> dict
    if is_main_process():
        logger = MMLogger.get_instance('test_wsi', log_file=infer_log_savepath)
        print(f"\n{'='*40}")
        print(f'Total patients: {len(data_list) + len(done_pidlist)}')
        print(f'Resumed {len(done_pidlist)}, Left {len(data_list)}.')
        print(f"{'='*40}")
    
    for ridx,row in enumerate(data_list):
        patientId = row["patientId"]
        if is_main_process():
            start_time = time.time()
        wsi_handler = WSIHandler(row["kfb_path"], PATCH_EDGE, LEVEL, positive_thr=POSITIVE_THR)
        slide_patchlist = wsi_handler.init_patchlist({
            'image': None,
            'valid_prob': 0, 
            'valid_flag': -1,
            'img_prob': 0, 
            'pred_label': -1,
        })
        # ---- 数据切分（保证每张卡处理的数据不重复） ----
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        data_per_rank = slide_patchlist[rank::world_size]
    
        valid_datapool, mlcls_datapool = [],[]
        for p_idx,patchinfo in enumerate(data_per_rank):
            img_input,_ = wsi_handler.read_cv2img(patchinfo['xy'])
            patchinfo['image'] = img_input
            valid_datapool.append(patchinfo)
            if len(valid_datapool) % test_bs == 0 or p_idx == len(data_per_rank)-1:
                wsi_handler.inference_valid_batch(valid_model, valid_datapool)
                mlcls_datapool = [item for item in valid_datapool if item['valid_flag']==2]
                if len(mlcls_datapool) > 0:
                    wsi_handler.inference_batch_pn(mlcls_model, mlcls_datapool)
                for item in valid_datapool:
                    del item['image']
                torch.cuda.empty_cache()
                valid_datapool, mlcls_datapool = [],[]
                print(f'\r[Rank {rank}] Processed {p_idx+1}/{len(data_per_rank)}', end='')
        
        all_results = [None for _ in range(dist.get_world_size())]
        torch.cuda.synchronize()    # 等当前 GPU 上的计算任务完成（防止 GPU 异步计算没结束）
        dist.all_gather_object(all_results, data_per_rank)
        dist.barrier()  # 等所有 rank 到达这里（防止 rank0 提前汇总）
        if dist.get_rank() == 0:    # rank0 汇总结果
            merged = [x for r in all_results for x in r]
            FP_patches = [i for i in merged if i['pred_label']==1]
            random.shuffle(FP_patches)
            infer_results = []
            for idx,patchinfo in enumerate(FP_patches[:FP_savenum]):
                filename = f'{patientId}_round{hs_round_idx}_{idx}.png'
                read_result = wsi_handler.read_PILimg(patchinfo['xy'])
                read_result.save(f'{FP_img_savedir}/{filename}')
                infer_results.append({
                    'patientId': patientId,
                    'filename': filename,
                    'square_coords': patchinfo['coords'],
                    'bboxes': [],
                    'clsnames': [],
                    'prefix': f'neg_slide_r{hs_round_idx}',
                    'diagnose': 0,
                    'maskfile': ''
                })

            # 打印当前切片的推理结果
            t_delta = time.time() - start_time
            logstr = wsi_handler.format_logstr(merged)
            logstr = f"[{patientId}] cost:{t_delta:0.2f}s, {logstr}"
            with open(f'{tmp_save_dir}/{patientId}.json', 'w', encoding='utf-8') as f:
                json.dump({
                    'log_str': logstr,
                    'neg_patch_list': infer_results,
                }, f, ensure_ascii=False)
            logger.info(f"({ridx+1}/{len(data_list)}){logstr}")


def get_models(device, gpu):
    init_default_scope('mmpretrain')
    valid_model = ValidClsNet()
    valid_model.to(device)
    valid_model.device = device
    valid_model.eval()
    valid_model.load_state_dict(torch.load(valid_ckpt))
    valid_model = torch.nn.parallel.DistributedDataParallel(
        valid_model, device_ids=[gpu], find_unused_parameters=False)
    valid_model = valid_model.module

    cfg = Config.fromfile(mmcls_config_file)
    cfg.backbone_cfg['backbone_ckpt'] = None
    mlcls_model = PatchNet(cfg).to(device)
    mlcls_model.img_size = cfg.input_size
    mlcls_model.load_ckpt(mmcls_ckpt)
    mlcls_model.eval()
    mlcls_model = torch.nn.parallel.DistributedDataParallel(
        mlcls_model, device_ids=[gpu], find_unused_parameters=False)
    mlcls_model = mlcls_model.module

    return valid_model,mlcls_model

def collect_tmp():
    total_patchlist,txt_lines = [],[]
    for filename in os.listdir(tmp_save_dir):
        with open(f'{tmp_save_dir}/{filename}', 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        total_patchlist.extend(json_data['neg_patch_list'])
        txt_lines.append(json_data['log_str']+'\n')
    with open(FP_json_savepath, 'w', encoding='utf-8') as f:
        json.dump(total_patchlist, f, ensure_ascii=False)
    txt_lines.append(f'Saved patches num: {len(total_patchlist)}.\n')
    with open(collect_result_savepath, 'w') as f:
        f.writelines(txt_lines)
        
    shutil.rmtree(tmp_save_dir)
    return len(total_patchlist)

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    init_distributed_mode(args)
    set_seed(SEED)
    device = torch.device(f'cuda:{os.getenv("LOCAL_RANK")}')
    valid_model,mlcls_model = get_models(device,args.gpu)
    os.makedirs(FP_img_savedir, exist_ok=True, mode=0o777)
    
    run_inference(valid_model, mlcls_model)
    torch.cuda.synchronize()    # 等当前 GPU 上的计算任务完成（防止 GPU 异步计算没结束）
    dist.barrier()  # 等所有 rank 到达这里（防止 rank0 提前汇总）
    if dist.get_rank() == 0:    # rank0 汇总结果
        patch_num = collect_tmp()
        print(f"\n{'='*40}")
        print(f'Saved patches num: {patch_num}.')
        print(f'JSON file saved in {FP_json_savepath}')
        print(f'Slide infer result saved in {infer_log_savepath}')
        print(f"{'='*40}")

    torch.distributed.destroy_process_group()

if __name__ == '__main__':
    main()



'''
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=12342 scripts/process_WSI/mlc_negslide_hs.py
'''