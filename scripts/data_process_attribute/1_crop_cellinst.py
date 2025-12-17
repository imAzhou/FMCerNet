import json
import torch
import os
import pandas as pd
from tqdm import tqdm
from cerwsi.utils import KFBSlide
from PIL import Image
import torch.distributed as dist
import argparse
from cerwsi.utils import set_seed, init_distributed_mode, is_main_process


def crop_cellinst(img_save_dir):
    JFSW_1_data = pd.read_csv('data_resource/group_csv/JFSW_1.csv')
    JFSW_2_data = pd.read_csv('data_resource/group_csv/JFSW_2.csv')
    JFSW_data = pd.concat([JFSW_1_data, JFSW_2_data])

    with open('data_resource/cell_attri/cell_inst.json', 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    pId_list = list(json_data.keys())
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    data_per_rank = pId_list[rank::world_size]
    
    if is_main_process():
        pbar = tqdm(data_per_rank, ncols=80)
    else:
        pbar = data_per_rank

    new_json_data = {}
    for patientId in pbar:
        celllist = json_data[patientId]
        new_json_data[patientId] = []
        rowInfo = JFSW_data[JFSW_data['patientId'] == patientId].iloc[0]
        slide = KFBSlide(rowInfo['kfb_path'])
        for idx,cellitem in enumerate(celllist):
            filename = f'{patientId}_cell{idx}.png'
            cellitem['filename'] = filename
            px1,py1,px2,py2 = cellitem['bbox']
            location, level, size = (px1,py1), 0, (px2-px1,py2-py1)
            patch_img = Image.fromarray(slide.read_region(location, level, size))
            patch_img.save(f'{img_save_dir}/{filename}')
            new_json_data[patientId].append(cellitem)
    
    all_results = [None for _ in range(dist.get_world_size())]
    torch.cuda.synchronize()    # 等当前 GPU 上的计算任务完成（防止 GPU 异步计算没结束）
    dist.all_gather_object(all_results, new_json_data)
    dist.barrier()  # 等所有 rank 到达这里（防止 rank0 提前汇总）
    if dist.get_rank() == 0:    # rank0 汇总结果
        merged = {}
        for i in all_results:
            merged.update(i)
        with open('data_resource/cell_attri/cell_inst_named.json', 'w', encoding='utf-8') as f:
            json.dump(merged, f, ensure_ascii=False)



def main():
    img_save_dir = 'data_resource/cell_attri/cell_inst/images'
    os.makedirs(img_save_dir, exist_ok=True, mode=0o777)
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    init_distributed_mode(args)
    set_seed(1234)
    
    crop_cellinst(img_save_dir)
    torch.cuda.synchronize()    # 等当前 GPU 上的计算任务完成（防止 GPU 异步计算没结束）
    dist.barrier()  # 等所有 rank 到达这里（防止 rank0 提前汇总）
    # if dist.get_rank() == 0:    # rank0 汇总结果
    #     # collect_tmp()
    #     print(f"\n{'='*40}")
    #     print(f'WSI infer result saved in {infer_txt_savepath}')
    #     print(f"{'='*40}")

    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()

'''
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=12341 scripts/data_process_attribute/1_crop_cellinst.py
'''