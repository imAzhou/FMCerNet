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
from mmdet.models.detectors import DINO
from types import SimpleNamespace
from cerwsi.nets.backbone.SmartCCS_backbone import SmartCCS
from cerwsi.utils import set_seed, init_distributed_mode, is_main_process
from cerwsi.utils.wsi_handler import WSIHandler
from mmengine.logging import MMLogger
import pandas as pd
import time
from multiprocessing import Pool
import copy
from torchvision import transforms


LEVEL,PATCH_EDGE = 0,1200
CERTAIN_THR,POSITIVE_THR = 0.7,0.5
SEED,SAFE_MARGIN = 1234,100
BBOX_CLASS = ['AGC', 'ASC-US', 'LSIL', 'ASC-H', 'HSIL']
valid_bs, detector_bs = 128, 8
valid_ckpt = 'checkpoints/valid_cls_best.pth'
WSI_feat_savedir = f'data_resource/0630/WINDOW_SIZE_{PATCH_EDGE}/slide_feat_detector'
os.makedirs(WSI_feat_savedir, exist_ok=True, mode=0o777)
infer_csv_files = [
    'data_resource/0630/45_0924_train.csv',
    'data_resource/0630/67_0924_val.csv',
    # 'data_resource/0630/speed_test.csv',
]

detector_rootdir = 'log/WS1200/dino_r50'
detector_config_file = f"{detector_rootdir}/vis_data/config.py"
detector_ckpt = f"{detector_rootdir}/epoch_2.pth"
infer_log_savepath = f'{detector_rootdir}/infer.log'
infer_txt_savepath = f'{detector_rootdir}/infer_result.txt'

tmp_dir = f'{detector_rootdir}/tmp'
tmp_logtxt_dir = f'{tmp_dir}/extract_WSI_topk'
os.makedirs(tmp_logtxt_dir, exist_ok=True, mode=0o777)


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

def run_inference(valid_model,extractor_model,cell_detector):
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
                                 certain_thr=CERTAIN_THR, positive_thr=POSITIVE_THR,
                                 positive_class=BBOX_CLASS)
        slide_patchlist = wsi_handler.init_patchlist({
            'image': None,
            'filepath': '',
            'valid_prob': 0, 
            'valid_flag': -1,
            'img_prob': 0, 
            'pred_label': -1,
            'img_token': None,
            'pred_bboxes': []
        })

        # ---- 数据切分（保证每张卡处理的数据不重复） ----
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        data_per_rank = slide_patchlist[rank::world_size]
        data_per_rank = load_patchimgs(row["kfb_path"], data_per_rank)

        for p_idx in range(0, len(data_per_rank), valid_bs):
            read_pool = data_per_rank[p_idx:p_idx+valid_bs]
            wsi_handler.infer_valid_fn(valid_model, read_pool)
        print(f'\r[Rank {rank}] Load {len(data_per_rank)} tiles.', end='')
        
        valid_list = [item for item in data_per_rank if item['valid_flag'] != 0]
        for p_idx in range(0, len(valid_list), detector_bs):
            read_pool = valid_list[p_idx:p_idx+detector_bs]
            wsi_handler.infer_celldetector_fn(cell_detector, read_pool)
            for item in read_pool:
                del item['image']
            torch.cuda.empty_cache()
        # clear invalid img
        for item in data_per_rank:
            if item['valid_flag'] == 0:
                del item['image']
        torch.cuda.empty_cache()
        
        lesion_list = [{
            'coord': bboxinfo['coord'], 
            'score': bboxinfo['score'],
            'lesion_token': None
        } for item in valid_list for bboxinfo in item['pred_bboxes']]
        print(f'\r[Rank {rank}] Detected {len(lesion_list)} lesions.', end='')

        transform = transforms.Compose([
            transforms.Resize((extractor_model.img_size, extractor_model.img_size)),   # resize 到指定大小
            transforms.ToTensor(),     # (H,W,C) [0,255] → (C,H,W) [0,1]
            transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
        ])
        for lesion_item in lesion_list:
            bx1, by1, bx2, by2 = lesion_item['coord']
            bboxwh = (bx2-bx1, by2-by1)
            pil_img = wsi_handler.read_PILimg((bx1, by1),bboxwh)
            tensor_img = transform(pil_img)
            inputs = tensor_img.unsqueeze(0).to(extractor_model.device)  # (1,3,h,w)
            with torch.no_grad():
                outputs = extractor_model(inputs)
            lesion_token = outputs['x_norm_clstoken'][0].detach().cpu()
            lesion_item['lesion_token'] = copy.deepcopy(lesion_token)
            del inputs,outputs
        torch.cuda.empty_cache()

        all_results = [None for _ in range(dist.get_world_size())]
        torch.cuda.synchronize()    # 等当前 GPU 上的计算任务完成（防止 GPU 异步计算没结束）
        dist.all_gather_object(all_results, lesion_list)
        dist.barrier()  # 等所有 rank 到达这里（防止 rank0 提前汇总）
        if dist.get_rank() == 0:    # rank0 汇总结果
            merged = [x for r in all_results for x in r]
            selected = sorted(
                merged,
                key=lambda x: x['score'], reverse=True
            )
            if len(selected) > 0:
                slide_feats = torch.stack([pinfo['lesion_token'] for pinfo in selected])
                torch.save(slide_feats, f"{WSI_feat_savedir}/{patientId}.pt")
            # 打印当前切片的推理结果
            t_delta = time.time() - start_time
            logstr = f'Detected {len(merged)} lesions.'
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

    model_cfg = {
        'backbone_cfg': {
            'use_peft': None,
            'frozen_backbone': False,
            'backbone_ckpt': 'checkpoints/CCS_vitl_100M.pth'
        }
    }
    cfg = SimpleNamespace(**model_cfg)
    extractor_model = SmartCCS(cfg).to(device)
    extractor_model.img_size = 224
    extractor_model.eval()
    extractor_model = torch.nn.parallel.DistributedDataParallel(
        extractor_model, device_ids=[gpu], find_unused_parameters=False)
    extractor_model = extractor_model.module

    init_default_scope('mmdet')
    cfg = Config.fromfile(detector_config_file)
    del cfg.model.type
    cell_detector = DINO(**cfg.model).to(device)
    cell_detector.device = device
    cell_detector.img_size = cfg.input_size
    cell_detector.eval()
    ckpt = torch.load(detector_ckpt, weights_only=False, map_location=device)['state_dict']
    cell_detector.load_state_dict(ckpt)
    cell_detector = torch.nn.parallel.DistributedDataParallel(
        cell_detector, device_ids=[gpu], find_unused_parameters=False)
    cell_detector = cell_detector.module

    return valid_model,extractor_model,cell_detector

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
    valid_model,extractor_model,cell_detector = get_models(device,args.gpu)
    
    run_inference(valid_model,extractor_model,cell_detector)
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
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=12346 scripts/process_WSI/extract_WSI_topk_detector.py
'''