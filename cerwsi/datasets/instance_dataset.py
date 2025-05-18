import torch
from PIL import Image
from torch.utils.data import Dataset
import json
import numpy as np
from mmcv.transforms import Compose
from collections import defaultdict
import pickle
import torch.nn.functional as F
import random

from tqdm import tqdm

# 自定义数据集类
class InstanceDataset(Dataset):
    def __init__(self, data_cfg, annojson_path, rle_masks_path, transform):
        """
        Args:
            img_dir (str): img dir
        """
        self.img_dir = data_cfg.img_dir
        self.instance_mask_dir = data_cfg.instance_mask_dir
        self.num_classes = data_cfg.num_classes
        self.classes = data_cfg.classes
        
        self.transform = Compose(transform)
        self.annojson_path = annojson_path
        with open(self.annojson_path, 'r') as f:
            self.patch_COCOinfo = json.load(f)

        self.palette = [[255,255,255] for i in range(len(self.classes))]
        for category in self.patch_COCOinfo['categories']:
            self.palette[category['id']] = category['color']
        
        with open(rle_masks_path, 'rb') as f:
            rle_masks = pickle.load(f)
        self.format_anno2COCO(rle_masks)

    def format_anno2COCO(self, rle_masks):
        print('Start to format annotaion in COCO style.')
        annoInimg = defaultdict(list)
        for annoinfo in self.patch_COCOinfo['annotations']:
            x,y,w,h = annoinfo['bbox']
            inst_mask = rle_masks[annoinfo['image_id']][annoinfo["inst_id"]]
            annoInimg[annoinfo['image_id']].append({
                'bbox': [x, y, x+w, y+h],
                'bbox_label': annoinfo['category_id'],
                'mask': inst_mask,
                'inst_id': annoinfo['inst_id'],
                'ignore_flag': False     # False means retain 
            })

        self.imginfo_list = []
        for imginfo in tqdm(self.patch_COCOinfo['images'], ncols=80):
            imginfo['img_path'] = f'{self.img_dir}/{imginfo["prefix"]}/{imginfo["file_name"]}'
            imginfo['img_id'] = imginfo['id']
            imginfo['instances'] = annoInimg[imginfo['img_id']]
            del imginfo['id']
            self.imginfo_list.append(imginfo)
        print('Done to format annotaion in COCO style.')

    def __len__(self):
        return len(self.patch_COCOinfo['images'])

    def __getitem__(self, idx):
        output = self.transform(self.imginfo_list[idx])

        output['data_samples'].diagnose = self.imginfo_list[idx]['diagnose']
        output['data_samples'].prefix = self.imginfo_list[idx]['prefix']

        return output
    
    def generate_instance_GT(self, imginfo):
        image_label = imginfo['diagnose']
        instance_mask, instance_label = [],[]
        if image_label != 0:
            purename = imginfo["filename"].split('.')[0]
            data = np.load(f'{self.instance_mask_dir}/{purename}.npz')
            instance_mask = data['masks']      # (n, h, w)
            instance_label = data['labels']    # (n,)
        return instance_mask, instance_label

    def generate_clsid_mask(self, imginfo, shape):
        w, h = shape
        image_label = imginfo['diagnose']
        multi_pos_label = torch.zeros((self.num_classes-1,), dtype=torch.float32)
        
        if image_label == 0:
            gt_mask = torch.ones((h,w), dtype=torch.int32)
        else:
            purename = imginfo["filename"].split('.')[0]
            data = np.load(f'{self.mask_dir}/{purename}.npz')
            nonzero_indices = data['indices']
            nonzero_values = data['values']  # 1代表阴性，>1 代表阳性
            shape = tuple(data['shape'])
            restored_gt_mask = np.zeros((h,w), dtype=int)
            restored_gt_mask[nonzero_indices[0], nonzero_indices[1]] = nonzero_values
            gt_mask = torch.as_tensor(restored_gt_mask).to(torch.int32)
            pos_label_list = [i-2 for i in list(set(nonzero_values))]   # [0,4]
            multi_pos_label[pos_label_list] = 1

        return gt_mask,multi_pos_label

    def generate_bbox_mask(self, bboxes, bboxes_clsid, shape):
        """
        生成一个形状为 shape 的矩阵，初始为 0，
        bbox 区域填充对应类别 ID，并优先填充小框的类别 ID。
        
        :param bboxes: List of bboxes [[x1, y1, x2, y2], ...]
        :param bboxes_clsid: List of clsid [1,3,...]
        :param shape: (height, width) 矩阵的目标尺寸
        :return: 生成的矩阵
        """
        h, w = shape
        mask = torch.zeros((h, w), dtype=torch.int32)
        
        # 按照 bbox 面积从大到小排序，确保小框后填充
        bboxes = sorted(bboxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
        
        for (x1, y1, x2, y2), class_id in zip(bboxes,bboxes_clsid):
            x1, y1, x2, y2 = int(max(0, x1)), int(max(0, y1)), int(min(w, x2)), int(min(h, y2))  # 限制边界
            mask[y1:y2, x1:x2] = class_id  # 填充类别 ID
        
        return mask
    