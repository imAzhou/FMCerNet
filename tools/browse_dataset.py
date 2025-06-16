# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import os
import argparse
from mmdet.models.utils import mask2ndarray
from mmdet.structures.bbox import BaseBoxes
from mmengine.config import Config
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import cv2
from cerwsi.utils import show_box,show_mask
from cerwsi.datasets import load_data
from mmdet.structures import DetDataSample


parser = argparse.ArgumentParser()

# base args
parser.add_argument('dataset_config_file', type=str)
parser.add_argument('save_dir', type=str)
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--vis_batch_nums', type=int, default=-1)

def draw_dataset_gt(img, data_sample: DetDataSample, save_path, metainfo):
    classes = metainfo['classes']
    palette = metainfo['palette']

    gt_mask = data_sample.gt_instances.masks
    gt_boxes = data_sample.gt_instances.bboxes
    boxes_clsids = data_sample.gt_instances.labels

    fig = plt.figure(figsize=(13,13))
    ax = fig.add_subplot(111)
    ax.imshow(img)

    for box,cls_id,mask in zip(gt_boxes, boxes_clsids, gt_mask): 
        cls_color = palette[cls_id]
        edgecolor = np.array([cls_color[0]/255, cls_color[1]/255, cls_color[2]/255, 1])
        show_box(box, ax, edgecolor=edgecolor)
        show_mask(mask, ax, rgb=cls_color)
        plt.contour(mask, levels=[0.5], colors='lime', linewidths=2)
        x1, y1 = box[:2]
        class_name = classes[cls_id]
        ax.text(x1, y1, class_name, fontsize=10, color='white',
                bbox=dict(facecolor=np.array(cls_color)/255., alpha=0.5, edgecolor='none'))

    ax.set_title('GT info')
    patches = [mpatches.Patch(facecolor=np.array(palette[i])/255., label=classes[i], edgecolor='black') for i in range(len(classes))]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='large')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True, mode=0o777)
    d_cfg = Config.fromfile(args.dataset_config_file)
    dataloader = load_data(d_cfg, ['val'])
    metainfo = {
        'classes': dataloader.dataset.classes,
        'palette': dataloader.dataset.palette,
    }
    for i_batch, sampled_batch in enumerate(tqdm(dataloader, ncols=70)):

        if args.vis_batch_nums > 0 and i_batch > args.vis_batch_nums:
            break
        
        for idx,data_sample in enumerate(sampled_batch['data_samples']):
            filename = data_sample.img_path.split('/')[-1]
            patientId = '_'.join(filename.split('_')[:3])
            if patientId != 'JFSW_2_302':
                continue
            if data_sample.diagnose == 0:
                continue

            img = sampled_batch['inputs'][idx].permute(1, 2, 0).int().numpy()
            img = img[..., [2, 1, 0]]  # bgr to rgb
            data_sample = data_sample.numpy()
            gt_instances = data_sample.gt_instances
            gt_bboxes = gt_instances.get('bboxes', None)
            if gt_bboxes is not None and isinstance(gt_bboxes, BaseBoxes):
                gt_instances.bboxes = gt_bboxes.tensor
            gt_masks = gt_instances.get('masks', None)
            if gt_masks is not None:
                masks = mask2ndarray(gt_masks)
                gt_instances.masks = masks.astype(bool)
            data_sample.gt_instances = gt_instances

            out_file = f'{args.save_dir}/{filename}'
            draw_dataset_gt(img, data_sample, out_file, metainfo)
            print(f'{filename} instances number: {len(gt_instances.bboxes)}')


if __name__ == '__main__':
    main()

'''
python tools/browse_dataset.py \
    configs/dataset/mmdet/l_cerscanv1_dataset.py \
    statistic_results/visual_results/gt_visual_512_val \
    --vis_batch_nums -1
'''