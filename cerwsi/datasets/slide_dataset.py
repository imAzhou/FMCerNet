import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import pandas as pd
import os

# 自定义数据集类
class SlideDataset(Dataset):
    def __init__(self, cfg, csvfile):
        df_data = pd.read_csv(csvfile)
        df_data = df_data.drop_duplicates(subset=["patientId"])   # 按 patientId 去重
        self.data_list = df_data.to_dict(orient="records")  # 每一行 -> dict
        self.feat_dir = cfg.feat_dir
        self.classes = cfg.classes
        self.cls_map = cfg.cls_map
        self.patch_nums = cfg.patch_nums
        self.C_in = cfg.C_in

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        slide_info = self.data_list[idx]
        patientId = slide_info['patientId']
        feat_path = f'{self.feat_dir}/{patientId}.pt'
        if os.path.exists(feat_path):
            slide_tensor = torch.load(feat_path)    # (L, dim)
            L, dim = slide_tensor.shape
            if L > self.patch_nums:
                slide_tensor = slide_tensor[:self.patch_nums, :]
            elif L < self.patch_nums:
                pad_size = (0, 0, 0, self.patch_nums - L)  
                # pad_size 含义 (dim2_pad_left, dim2_pad_right, dim1_pad_left, dim1_pad_right)
                # 这里在第 0 维（序列长度）末尾补 (target_len - L) 行
                slide_tensor = F.pad(slide_tensor, pad_size, value=0)
        else:
            slide_tensor = torch.zeros(self.patch_nums, self.C_in)

        # slide_tensor = torch.rand(self.patch_nums, 517)
        slide_clsname = slide_info['kfb_clsname']
        if self.cls_map is not None:
            slide_clsname = self.cls_map[slide_info['kfb_clsname']]
        slide_label = self.classes.index(slide_clsname)
        return {
            'inputs': slide_tensor,
            'data_samples': {
                'slide_label': slide_label,
                'slide_info': slide_info
            }
        }
