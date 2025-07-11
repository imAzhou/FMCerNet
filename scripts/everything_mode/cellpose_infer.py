import json
import numpy as np
from cellpose import models, core, io, plot, utils, transforms
from tqdm import tqdm
import cv2
import os
import torch

io.logger_setup() # run this to get printing of progress

model = models.CellposeModel(gpu=True, 
                                #  pretrained_model='/x22201018/.cellpose/models/cpsam',
                                 device=torch.device("cuda:0"))
diameter = 120

with open('data_resource/HMCHH/annofiles_roi/fold1_val.json', 'r', encoding='utf-8') as f:
    json_data = json.load(f)
proposal_savedir = f'data_resource/HMCHH/proposal_d{diameter}'
os.makedirs(proposal_savedir, exist_ok=True, mode=0o777)

for imgitem in tqdm(json_data['images'], ncols=80):
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


