import json
import torch
import os
import shutil
import warnings
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
warnings.filterwarnings("ignore", category=UserWarning)
import torch.distributed as dist
import argparse
import cv2
from mmengine.registry import init_default_scope
from cerwsi.nets import ValidClsNet
from cerwsi.utils import set_seed, init_distributed_mode, is_main_process
from cerwsi.utils.wsi_handler import WSIHandler
from mmpretrain.structures import DataSample
import pandas as pd

LEVEL,WINDOW_SIZE = 0,1600
CERTAIN_THR,POSITIVE_THR = 0.7,0.5
SEED,SAFE_MARGIN = 1234,100
valid_ckpt = 'checkpoints/valid_cls_best.pth'
infer_csv_files = [
    'data_resource/0630/4_pure_train.csv',
    'data_resource/0630/5_jfsw_train.csv',
    'data_resource/0630/6_val.csv',
]
data_root = f'data_resource/0630/WINDOW_SIZE_{WINDOW_SIZE}'
prefix = 'neg_slide_rc'
neg_slide_img_savedir = f'{data_root}/images/{prefix}'
neg_slide_json_savepath = f'{data_root}/ann_jsons/randomcut_in_negslide.json'
os.makedirs(neg_slide_img_savedir, exist_ok=True, mode=0o777)
neg_patch_thr,max_try = 5,100   # 约束：neg_patch_thr <= max_try
task_tag = 'random_cut_negslide'
infer_txt_savepath = f'log/{task_tag}/infer_result.txt'
tmp_save_dir = f'log/{task_tag}/tmp'
os.makedirs(tmp_save_dir, exist_ok=True, mode=0o777)


def collect_slidelist():
    done_pidlist = []
    for tmpfilename in os.listdir(tmp_save_dir):
        pid = tmpfilename.split('.')[0]
        done_pidlist.append(pid)
    all_df = []
    for csvfile in infer_csv_files:
        df = pd.read_csv(csvfile)
        df = df[df['kfb_clsid']==0]
        df = df.drop_duplicates(subset=["patientId"])   # 按 patientId 去重
        df = df[~df["patientId"].isin(done_pidlist)]
        all_df.append(df)
    cat_df = pd.concat(all_df)
    data_list = cat_df.to_dict(orient="records")  # 每一行 -> dict
    if is_main_process():
        print(f"\n{'='*40}")
        print(f'Total patients: {len(data_list) + len(done_pidlist)}')
        print(f'Resumed {len(done_pidlist)}, Left {len(data_list)}.')
        print(f"{'='*40}")
    return data_list

def run_inference(valid_model):
    data_list = collect_slidelist()
    # ---- 数据切分（保证每张卡处理的数据不重复） ----
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    data_per_rank = data_list[rank::world_size]
    
    for ridx,row in enumerate(data_per_rank):
        patientId = row['patientId']
        wsi_handler = WSIHandler(row["kfb_path"], WINDOW_SIZE)
        slide_patch_cnt, try_cnt = 0, 0
        neg_patch_list = []
        while slide_patch_cnt < neg_patch_thr and try_cnt < max_try:
            data_batch = dict(inputs=[], data_samples=[])
            img_input, coords = wsi_handler.read_cv2img(random_cut=True)
            img_input = torch.as_tensor(cv2.resize(img_input, (224,224)))
            data_batch['inputs'].append(img_input.permute(2,0,1))    # (bs, 3, h, w)
            data_batch['data_samples'].append(DataSample())
            data_batch['inputs'] = torch.stack(data_batch['inputs'], dim=0)
            with torch.no_grad():
                outputs = valid_model.val_step(data_batch)
        
            if max(outputs[0].pred_score) > CERTAIN_THR and outputs[0].pred_label == 1:
                filename = f'{patientId}_rc_{slide_patch_cnt}.png'
                neg_patch_list.append({
                    'patientId': patientId,
                    'filename': filename,
                    'square_coords': coords,    # 在媒体资源中的相对坐标
                    'bboxes': [],
                    'clsnames': [],
                    'prefix': prefix,
                    'diagnose': 0,
                    'maskfile': ''
                })
                read_result = wsi_handler.read_PILimg((coords[0], coords[1]))
                read_result.save(f'{neg_slide_img_savedir}/{filename}')
                slide_patch_cnt += 1
            try_cnt += 1
        
        print(f"\r[Rank {rank}] Processing {ridx}/{len(data_per_rank)} samples.\t", end='')
        if(slide_patch_cnt != neg_patch_thr):
            print(f"[Rank {rank}] slide_patch_cnt={slide_patch_cnt}, try_cnt={try_cnt}, patientId = {patientId}")
        with open(f'{tmp_save_dir}/{patientId}.json', 'w', encoding='utf-8') as f:
            json.dump({
                'log_str': f"[patientId {patientId}] slide_patch_cnt={slide_patch_cnt}, try_cnt={try_cnt}\n",
                'neg_patch_list': neg_patch_list,
            }, f, ensure_ascii=False)

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

    return valid_model

def collect_tmp():
    total_patchlist,txt_lines = [],[]
    for filename in os.listdir(tmp_save_dir):
        with open(f'{tmp_save_dir}/{filename}', 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        total_patchlist.extend(json_data['neg_patch_list'])
        txt_lines.append(json_data['log_str'])
    with open(neg_slide_json_savepath, 'w', encoding='utf-8') as f:
        json.dump(total_patchlist, f, ensure_ascii=False)
    txt_lines.append(f'Saved patches num: {len(total_patchlist)}.\n')
    with open(infer_txt_savepath, 'w') as f:
        f.writelines(txt_lines)
    shutil.rmtree(tmp_save_dir)
    return len(total_patchlist)

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    init_distributed_mode(args)
    set_seed(SEED)
    device = torch.device(f'cuda:{os.getenv("LOCAL_RANK")}')
    valid_model = get_models(device,args.gpu)
    
    run_inference(valid_model)
    torch.cuda.synchronize()    # 等当前 GPU 上的计算任务完成（防止 GPU 异步计算没结束）
    dist.barrier()  # 等所有 rank 到达这里（防止 rank0 提前汇总）
    if dist.get_rank() == 0:    # rank0 汇总结果
        patch_num = collect_tmp()
        print(f"\n{'='*40}")
        print(f'Saved patches num: {patch_num}.')
        print(f'JSON file saved in {neg_slide_json_savepath}')
        print(f'Slide infer result saved in {infer_txt_savepath}')
        print(f"{'='*40}")

    torch.distributed.destroy_process_group()

if __name__ == '__main__':
    main()
    


'''
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=12346 scripts/process_WSI/random_cut_negslide.py
'''