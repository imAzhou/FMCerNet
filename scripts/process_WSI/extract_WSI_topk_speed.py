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
import time
from multiprocessing import Pool


LEVEL,PATCH_EDGE = 0,1200
CERTAIN_THR,POSITIVE_THR = 0.7,0.5
SEED,SAFE_MARGIN = 1234,100
test_bs = 64
valid_ckpt = 'checkpoints/valid_cls_best.pth'
WSI_feat_savedir = f'data_resource/0630/WINDOW_SIZE_{PATCH_EDGE}/slide_feat_ours'
os.makedirs(WSI_feat_savedir, exist_ok=True, mode=0o777)
infer_csv_files = [
    # 'data_resource/0630/45_0924_train.csv',
    # 'data_resource/0630/67_0924_val.csv',
    'data_resource/0630/reinfer.csv',
]

pnmodel_rootdir = 'log/WS1200/wscernet'
mmcls_config_file = f'{pnmodel_rootdir}/config.py'
mmcls_ckpt = f'{pnmodel_rootdir}/checkpoints/epoch_19.pth'
infer_log_savepath = f'{pnmodel_rootdir}/infer.log'
infer_txt_savepath = f'{pnmodel_rootdir}/infer_result_speed.txt'

tmp_dir = f'{pnmodel_rootdir}/tmp'
tmp_logtxt_dir = f'{tmp_dir}/extract_WSI_topk'
os.makedirs(tmp_logtxt_dir, exist_ok=True, mode=0o777)


'''
tmp_logtxt_dir
    - patientId.txt: log str
'''

def resume_infer():
    done_pidlist = []
    for txtname in os.listdir(tmp_logtxt_dir):
        pid = txtname.split('.')[0]
        done_pidlist.append(pid)
    return done_pidlist

def readimgs(proc_id, kfb_path, set_group):
    wsi_handler = WSIHandler(kfb_path, PATCH_EDGE, LEVEL, 
                certain_thr=CERTAIN_THR, positive_thr=POSITIVE_THR)
    for item in set_group:
        item['image'],_ = wsi_handler.read_cv2img(item['xy'])
    return set_group

def load_patchimgs(kfb_path, slide_patchlist):
    cpu_num = 16
    step = len(slide_patchlist) // cpu_num
    workers = Pool(processes=cpu_num)
    processes = []
    for proc_id in range(cpu_num):
        if proc_id == cpu_num-1:
            set_group = slide_patchlist[proc_id*step:]
        else:
            set_group = slide_patchlist[proc_id*step:(proc_id+1)*step]
        p = workers.apply_async(readimgs,(proc_id, kfb_path, set_group))
        processes.append(p)
    readresults = []
    for p in processes:
        results = p.get()
        readresults.extend(results)
    return readresults

def run_inference(valid_model, mlcls_model):
    done_pidlist = resume_infer()
    total_datalist = []
    for csv_file in infer_csv_files:
        df = pd.read_csv(csv_file)
        df = df.drop_duplicates(subset=["patientId"])   # 按 patientId 去重
        df = df[~df["patientId"].isin(done_pidlist)]
        data_list = df.to_dict(orient="records")  # 每一行 -> dict
        total_datalist.extend(data_list)
    # total_datalist = total_datalist[:2]
    if is_main_process():
        logger = MMLogger.get_instance('test_wsi', log_file=infer_log_savepath)
        print(f"\n{'='*40}")
        print(f'Total patients: {len(total_datalist) + len(done_pidlist)}')
        print(f'Resumed {len(done_pidlist)}, Left {len(total_datalist)}.')
        print(f"{'='*40}")
    
    for ridx,row in enumerate(total_datalist):
        patientId = row["patientId"]
        slide_clsname = row["kfb_clsname"]
        if is_main_process():
            start_time = time.time()
        wsi_handler = WSIHandler(row["kfb_path"], PATCH_EDGE, LEVEL, 
                                 certain_thr=CERTAIN_THR, positive_thr=POSITIVE_THR)
        slide_patchlist = wsi_handler.init_patchlist({
            'image': None,
            'filepath': '',
            'valid_prob': 0, 
            'valid_flag': -1,
            'img_prob': 0, 
            'pred_label': -1,
            'img_token': None
        })

        # ---- 数据切分（保证每张卡处理的数据不重复） ----
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        data_per_rank = slide_patchlist[rank::world_size]
        data_per_rank = load_patchimgs(row["kfb_path"], data_per_rank)

        for p_idx in range(0, len(data_per_rank), test_bs):
            read_pool = data_per_rank[p_idx:p_idx+test_bs]
            wsi_handler.infer_valid_fn(valid_model, read_pool)
            print(f'\r[Rank {rank}] Processed {p_idx+1}/{len(data_per_rank)} ', end='')

        wsi_handler.infer_pn_batch_fn(mlcls_model, data_per_rank, test_bs)
        all_results = [None for _ in range(dist.get_world_size())]
        torch.cuda.synchronize()    # 等当前 GPU 上的计算任务完成（防止 GPU 异步计算没结束）
        dist.all_gather_object(all_results, data_per_rank)
        dist.barrier()  # 等所有 rank 到达这里（防止 rank0 提前汇总）
        if dist.get_rank() == 0:    # rank0 汇总结果
            merged = [x for r in all_results for x in r]
            # 取 img_prob > 0 的 tile, 即有效 tile
            selected = sorted(
                [v for v in merged if v['img_prob'] > 0],
                key=lambda x: x['img_prob'], reverse=True
            )
            if len(selected) > 0:
                slide_feats = torch.stack([pinfo['img_token'] for pinfo in selected])
                torch.save(slide_feats, f"{WSI_feat_savedir}/{patientId}.pt")
            # 打印当前切片的推理结果
            t_delta = time.time() - start_time
            logstr = wsi_handler.format_logstr(merged)
            logstr = f"[{patientId}]({slide_clsname}) cost:{t_delta:0.2f}s, {logstr}"
            with open(f'{tmp_logtxt_dir}/{patientId}.txt', 'w', encoding='utf-8') as f:
                f.write(logstr)
            logger.info(f"({ridx+1}/{len(total_datalist)}){logstr}")


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
    total_lines = []
    for filename in os.listdir(tmp_logtxt_dir):
        with open(f'{tmp_logtxt_dir}/{filename}', 'r') as f:
            read_str = f.readline()
            total_lines.append(f'{read_str}\n')
    with open(infer_txt_savepath, 'w') as f: 
        f.writelines(total_lines)
    shutil.rmtree(tmp_dir)

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    init_distributed_mode(args)
    set_seed(SEED)
    device = torch.device(f'cuda:{os.getenv("LOCAL_RANK")}')
    valid_model,mlcls_model = get_models(device,args.gpu)
    
    run_inference(valid_model, mlcls_model)
    torch.cuda.synchronize()    # 等当前 GPU 上的计算任务完成（防止 GPU 异步计算没结束）
    dist.barrier()  # 等所有 rank 到达这里（防止 rank0 提前汇总）
    if dist.get_rank() == 0:    # rank0 汇总结果
        collect_tmp()
        print(f"\n{'='*40}")
        print(f'WSI infer result saved in {infer_txt_savepath}')
        print(f"{'='*40}")

    torch.distributed.destroy_process_group()

if __name__ == '__main__':
    main()
    


'''
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node=7 --master_port=12341 scripts/process_WSI/extract_WSI_topk_speed.py
'''