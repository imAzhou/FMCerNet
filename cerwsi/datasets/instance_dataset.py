import torch
from PIL import Image
from torch.utils.data import Dataset
import json
import numpy as np
import torch.nn.functional as F
import random

# 自定义数据集类
class InstanceDataset(Dataset):
    def __init__(self, root_dir, annojson_path, transform, classes):
        """
        Args:
            img_dir (str): img dir
        """
        self.img_dir = f'{root_dir}/images'
        self.mask_dir = f'{root_dir}/mask'
        self.instance_mask_dir = f'{root_dir}/instance_mask'
        self.root_dir = root_dir
        self.annofiles_dir = f'{root_dir}/annofiles'
        with open(f'{self.annofiles_dir}/{annojson_path}', 'r') as f:
            self.patch_infolist = json.load(f)
            # self.patch_infolist = self.patch_infolist[:10000]
        
        self.transform = transform
        self.num_classes = len(classes)
        self.classes = classes

    def __len__(self):
        return len(self.patch_infolist)

    def __getitem__(self, idx):
        imginfo = self.patch_infolist[idx]
        imginfo['clsids'] = [self.classes.index(clsname) for clsname in imginfo['clsnames']]
        
        imgpath = f'{self.img_dir}/{imginfo["prefix"]}/{imginfo["filename"]}'
        imginfo['imgpath'] = imgpath
        image = Image.open(imgpath)
        imginfo['origin_size'] = (image.size[1], image.size[0])     # (H,W)
        input_tensor = self.transform(image)
        image_label = imginfo['diagnose']
        
        # imginfo['use_inst'] = False
        # if imginfo['prefix'] == 'total_pos':
        #     imginfo['use_inst'] = True
        # elif imginfo['prefix'] == 'neg' and random.random() < 0.2:
        #     imginfo['use_inst'] = True

        instance_mask, instance_label = self.generate_instance_GT(imginfo)
        return input_tensor,image_label,instance_mask, instance_label, imginfo
    
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
    