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
import pandas as pd
import time

LEVEL,PATCH_EDGE = 0,850
CERTAIN_THR,POSITIVE_THR = 0.7,0.5
SEED,SAFE_MARGIN = 1234,100
test_bs,topk_num = 64,100
valid_ckpt = 'checkpoints/valid_cls_best.pth'
WSI_feat_savedir = f'data_resource/0630/WINDOW_SIZE_{PATCH_EDGE}/slide_feat'
os.makedirs(WSI_feat_savedir, exist_ok=True, mode=0o777)
infer_csv_file = 'data_resource/0630/4_pure_train.csv'

pnmodel_rootdir = 'log/WS850/hs_round0'
mmcls_config_file = f'{pnmodel_rootdir}/config.py'
mmcls_ckpt = f'{pnmodel_rootdir}/checkpoints/best.pth'
infer_txt_savepath = f'{pnmodel_rootdir}/infer_result.txt'
tmp_save_dir = f'{pnmodel_rootdir}/tmp/extract_WSI_topk'
os.makedirs(tmp_save_dir, exist_ok=True, mode=0o777)

def resume_infer():
    done_pidlist = []
    for txtname in os.listdir(tmp_save_dir):
        pid = txtname.split('.')[0]
        done_pidlist.append(pid)
    return done_pidlist

def run_inference(valid_model, mlcls_model, resume):
    done_pidlist = []
    if resume:
        done_pidlist = resume_infer()

    df = pd.read_csv(infer_csv_file)
    df = df.drop_duplicates(subset=["patientId"])   # 按 patientId 去重
    data_list = df.to_dict(orient="records")  # 每一行 -> dict
    if is_main_process():
        print(f"\n{'='*40}")
        print(f'Total patients: {len(data_list)}')
        if resume:
            print(f'Resumed {len(done_pidlist)}, Left {len(data_list)-len(done_pidlist)}.')
        print(f"{'='*40}")

    # data_list = data_list[:10]
    # ---- 数据切分（保证每张卡处理的数据不重复） ----
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    data_per_rank = data_list[rank::world_size]

    for ridx,row in enumerate(data_per_rank):
        start_time = time.time()
        patientId,slide_clsname = row["patientId"],row["kfb_clsname"]
        if patientId in done_pidlist:
            continue
        wsi_handler = WSIHandler(PATCH_EDGE, LEVEL, row["kfb_path"])
        slide_patchlist = wsi_handler.init_patchlist({
            'image': None,
            'valid_prob': 0, 
            'valid_flag': -1,
            'img_prob': 0, 
            'pred_label': -1,
            'img_token': None
        })

        # slide_patchlist = slide_patchlist[:100]
        valid_datapool, mlcls_datapool = [],[]
        for p_idx,patchinfo in enumerate(slide_patchlist):
            patchinfo['image'] = wsi_handler.read_cv2img(patchinfo['xy'])
            valid_datapool.append(patchinfo)
            if len(valid_datapool) % test_bs == 0 or p_idx == len(slide_patchlist)-1:
                wsi_handler.inference_valid_batch(valid_model, valid_datapool)
                mlcls_datapool = [item for item in valid_datapool if item['valid_flag']==2]
                if len(mlcls_datapool) > 0:
                    wsi_handler.inference_batch_pn(mlcls_model, mlcls_datapool)
                for item in valid_datapool:
                    del item['image']
                torch.cuda.empty_cache()
                valid_datapool, mlcls_datapool = [],[]
                print(f'\r[Rank {rank}] Processed {p_idx+1}/{len(slide_patchlist)}', end='')

        # 先取 img_prob > 0 的前 topk
        selected = sorted(
            [v for v in slide_patchlist if v['img_prob'] > 0],
            key=lambda x: x['img_prob'], reverse=True
        )[:topk_num]
        # 不足时用 valid_prob 补足
        if len(selected) < topk_num:
            selected_invalid = sorted(
                [v for v in slide_patchlist if v['valid_flag'] != 2],
                key=lambda x: x['valid_prob'], reverse=True
            )[:topk_num - len(selected)]
            mlcls_datapool = []
            for sel_idx,patchinfo in enumerate(selected_invalid):
                patchinfo['image'] = wsi_handler.read_cv2img(patchinfo['xy'])
                mlcls_datapool.append(patchinfo)
                if len(mlcls_datapool) % test_bs == 0 or sel_idx == len(selected_invalid)-1:
                    wsi_handler.inference_batch_pn(mlcls_model, mlcls_datapool)
                    for item in mlcls_datapool:
                        del item['image']
                    torch.cuda.empty_cache()
                    mlcls_datapool = []

            selected += selected_invalid
        slide_feats = torch.stack([pinfo['img_token'] for pinfo in selected])
        torch.save(slide_feats, f"{WSI_feat_savedir}/{patientId}.pt")
        # 打印当前切片的推理结果
        logstr = wsi_handler.format_logstr(slide_patchlist)
        t_delta = time.time() - start_time
        with open(f'{tmp_save_dir}/{patientId}.txt', 'w', encoding='utf-8') as f:
            f.write(f'[{patientId}({slide_clsname})] cost:{t_delta:0.2f}s, ' + logstr)
        print(f'\t[Rank {rank}] ({ridx+1}/{len(data_per_rank)}) Processed {len(slide_patchlist)} patches.')

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
    for filename in os.listdir(tmp_save_dir):
        with open(f'{tmp_save_dir}/{filename}', 'r') as f:
            read_str = f.readline()
            total_lines.append(f'{read_str}\n')
    with open(infer_txt_savepath, 'w') as f: 
        f.writelines(total_lines)
    shutil.rmtree(tmp_save_dir)

def main(resume):
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    init_distributed_mode(args)
    set_seed(SEED)
    device = torch.device(f'cuda:{os.getenv("LOCAL_RANK")}')
    valid_model,mlcls_model = get_models(device,args.gpu)
    
    run_inference(valid_model, mlcls_model, resume)
    torch.cuda.synchronize()    # 等当前 GPU 上的计算任务完成（防止 GPU 异步计算没结束）
    dist.barrier()  # 等所有 rank 到达这里（防止 rank0 提前汇总）
    if dist.get_rank() == 0:    # rank0 汇总结果
        collect_tmp()
        print(f"\n{'='*40}")
        print(f'WSI infer result saved in {infer_txt_savepath}')
        print(f"{'='*40}")

    torch.distributed.destroy_process_group()

if __name__ == '__main__':
    main(resume=True)
    


'''
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node=7 --master_port=12341 scripts/process_WSI/extract_WSI_topk.py
'''