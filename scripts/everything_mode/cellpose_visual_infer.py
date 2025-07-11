import json
from tqdm import tqdm
import torch
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
# from pycocotools.coco import COCO

def draw_boxes(img, bboxes, color=(0, 255, 0)):
    for box in bboxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    return img

def main():
    jsonfile = 'data_resource/HMCHH/annofiles_roi/fold1_train.json'
    img_root = 'data_resource/HMCHH/JPEGImages/'
    visual_saveroot = 'statistic_results/cellpose_infer/hmchh'
    os.makedirs(visual_saveroot, exist_ok=True, mode=0o777)
    with open(jsonfile, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    coco = COCO(jsonfile)
    
    for imgitem in tqdm(json_data['images'], ncols=80):
        purename = imgitem["file_name"].split('.')[0]
        img_id = imgitem['id']
        img_path = f'{img_root}/{imgitem["file_name"]}'
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gt_boxes = [ann['bbox'] for ann in coco.loadAnns(coco.getAnnIds(imgIds=img_id))]
        # 将xywh转为x1y1x2y2
        gt_boxes = [[x, y, x + w, y + h] for x, y, w, h in gt_boxes]
        
        preds = {}
        for dtag in ['d150']:
            predfile = f'data_resource/HMCHH/proposal_{dtag}/{purename}.json'
            preds[dtag] = []
            if os.path.exists(predfile):
                with open(predfile, 'r', encoding='utf-8') as f:
                    preds[dtag] = json.load(f)
        
        fig, axs = plt.subplots(1, 2, figsize=(12, 7))
        titles = ['GT', 'd150']
        imgs = [
            draw_boxes(img.copy(), gt_boxes, color=(255, 0, 0)),
            draw_boxes(img.copy(), preds['d150'], color=(0, 255, 0)) if preds['d150'] else img.copy(),
            # draw_boxes(img.copy(), preds['d180'], color=(0, 0, 255)) if preds['d180'] else img.copy(),
        ]
        for ax, im, title in zip(axs, imgs, titles):
            ax.imshow(im)
            ax.set_title(title)
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(f'{visual_saveroot}/{purename}.png')
        plt.close()

def find_imgitem(purename, json_data):
    for imgitem in json_data['images']:
        if imgitem['file_name'] == f'{purename}.png':
            return imgitem

def find_imganns(imgitem, json_data):
    imgbboxes = []
    for annitem in json_data['annotations']:
        if annitem['image_id'] == imgitem['id']:
            x1,y1,w,h = annitem['bbox']
            imgbboxes.append([x1, y1, x1+w, y1+h])
    return imgbboxes

def test_demo():
    from cellpose import models,utils
    model = models.CellposeModel(gpu=True, 
                                 pretrained_model='/x22201018/.cellpose/models/cpsam',
                                 device=torch.device("cuda:1"))
    diameter = 120
    with open('data_resource/HMCHH/annofiles_roi/fold1_train.json', 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    purenames = ['1657bj008_0001']
    for purename in purenames:
        img_url = f'data_resource/HMCHH/JPEGImages/{purename}.png'
        imgitem = find_imgitem(purename, json_data)
        gt_bboxes = find_imganns(imgitem, json_data)
        img = cv2.imread(img_url)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        masks_pred, flows, styles = model.eval([img], 
                                            # niter=4000,     # 根据 cellprob & 预测距离计算 cell 需要的迭代次数
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

        outlines = utils.masks_to_outlines(masks_pred)
        outX, outY = np.nonzero(outlines)
        
        fig, axs = plt.subplots(1, 2, figsize=(12, 7))
        im = draw_boxes(img.copy(), gt_bboxes, color=(255, 0, 0))
        axs[0].imshow(im)
        axs[0].set_title('GT')
        axs[0].axis('off')

        im = draw_boxes(img.copy(), bboxes, color=(0, 255, 0)) if bboxes else img.copy()
        im[outX, outY] = np.array([0, 255, 0])
        axs[1].imshow(im)
        axs[1].set_title(f'd{diameter}')
        axs[1].axis('off')

        plt.tight_layout()
        visual_saveroot = 'statistic_results/cellpose_infer/hmchh'
        os.makedirs(visual_saveroot, exist_ok=True, mode=0o777)
        plt.savefig(f'{visual_saveroot}/{purename}.png')
        plt.close()

if __name__ == "__main__":
    # main()
    test_demo()

