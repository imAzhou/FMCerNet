import json
import numpy as np
from cellpose import models, core, io, plot, utils, transforms
from tqdm import tqdm
import cv2
import os
import torch
import multiprocessing
from multiprocessing import Pool

io.logger_setup() # run this to get printing of progress

diameter = 60

def infer_fn(proc_id, imgitems, proposal_savedir):
    model = models.CellposeModel(gpu=True, 
                                 pretrained_model='/x22201018/.cellpose/models/cpsam',
                                 device=torch.device("cuda:0"))

    for idx, imgitem in enumerate(imgitems):
        img_url = f'data_resource/HMCHH/JPEGImages/{imgitem["file_name"]}'
        img = cv2.imread(img_url)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        masks_pred, flows, styles = model.eval([img], 
                                            # niter=1000,     # 根据 cellprob & 预测距离计算 cell 需要的迭代次数
                                            batch_size=64,
                                            max_size_fraction=1,
                                            diameter=float(diameter)) # using more iterations for bacteria
        masks_pred = masks_pred[0]
        # 输出每个区域的外包围框坐标 (x1, y1, x2, y2)
        bboxes = []
        labels = np.unique(masks_pred)
        labels = labels[labels != 0]  # 排除背景 label 0
        for label in labels:
            ys, xs = np.where(masks_pred == label)
            y1, y2 = ys.min(), ys.max()
            x1, x2 = xs.min(), xs.max()
            bboxes.append([x1, y1, x2 + 1, y2 + 1])  # 注意：右边是非包含式，+1 保持一致性

        bboxes = np.array(bboxes).tolist()
        purename = imgitem["file_name"].split('.')[0]
        with open(f'{proposal_savedir}/{purename}.json', 'w', encoding='utf-8') as f:
            json.dump(bboxes, f, ensure_ascii=False)
        print(f'Core {proc_id} processed : {idx+1}/{len(imgitems)}.')


if __name__ == "__main__":
    with open('data_resource/HMCHH/annofiles_roi/fold1_train.json', 'r', encoding='utf-8') as f:
        json_data_train = json.load(f)
    with open('data_resource/HMCHH/annofiles_roi/fold1_val.json', 'r', encoding='utf-8') as f:
        json_data_val = json.load(f)
    
    proposal_savedir = f'data_resource/HMCHH/proposal_d{diameter}'
    os.makedirs(proposal_savedir, exist_ok=True, mode=0o777)

    total_all_imgitems = [
        *json_data_train['images'],
        *json_data_val['images'],
    ]

    all_imgitems = []
    for idx, imgitem in enumerate(total_all_imgitems):
        purename = imgitem["file_name"].split('.')[0]
        if not os.path.exists(f'{proposal_savedir}/{purename}.json'):
            all_imgitems.append(imgitem)
    infer_fn(0, all_imgitems, proposal_savedir)

    # cpu_num = 8
    # set_split = np.array_split(range(len(all_imgitems)), cpu_num)
    # print(f"Number of cores: {cpu_num}, set number of per core: {len(set_split[0])}")
    # multiprocessing.set_start_method('spawn', force=True)
    # workers = Pool(processes=cpu_num)
    # processes = []
    # for proc_id, set_group in enumerate(set_split):
    #     process_group = [all_imgitems[i] for i in set_group]
    #     p = workers.apply_async(infer_fn, (proc_id, process_group, proposal_savedir))
    #     processes.append(p)
    # for p in processes:
    #     p.get()


