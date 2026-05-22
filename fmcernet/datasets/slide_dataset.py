import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import pandas as pd
import os
from mmpretrain.structures import DataSample

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
        self.C_in = cfg.in_dim
        self.format_type = cfg.format_type

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        slide_info = self.data_list[idx]
        patientId = slide_info['patientId']
        feat_path = f'{self.feat_dir}/{patientId}.pt'

        slide_tensor = torch.zeros(self.patch_nums, self.C_in)
        # where 1 indicates valid positions and 0 indicates masked positions.
        attn_mask = torch.zeros(self.patch_nums,)

        if os.path.exists(feat_path):
            if self.format_type == 'direct':
                slide_tensor = self.format_slide_tensor_simple(feat_path)
            elif self.format_type == 'pn_pos':
                slide_tensor = self.format_slide_tensor(feat_path)
            elif self.format_type == 'nopn_pos':
                slide_tensor = self.format_slide_tensor_nopn(feat_path)

        L, dim = slide_tensor.shape
        attn_mask[:L] = 1
        if L > self.patch_nums:
            slide_tensor = slide_tensor[:self.patch_nums, :]
        elif L < self.patch_nums:
            pad_size = (0, 0, 0, self.patch_nums - L)  
            # pad_size 含义 (dim2_pad_left, dim2_pad_right, dim1_pad_left, dim1_pad_right)
            # 这里在第 0 维（序列长度）末尾补 (target_len - L) 行
            slide_tensor = F.pad(slide_tensor, pad_size, value=0)

        # slide_tensor = torch.rand(self.patch_nums, 517)
        slide_clsname = slide_info['kfb_clsname']
        if self.cls_map is not None:
            slide_clsname = self.cls_map[slide_info['kfb_clsname']]
        slide_label = self.classes.index(slide_clsname)
        data_samples = DataSample()
        data_samples.slide_label = slide_label
        data_samples.slide_info = slide_info
        data_samples.attn_mask = attn_mask
        return {
            'inputs': slide_tensor,
            'data_samples': data_samples
        }
    
    def format_slide_tensor(self, feat_path):
        '''适用于 ours 的方法'''
        load_tensor = torch.load(feat_path)    # (L, dim)
        pn_prob_feat = load_tensor[:,0,:]
        pn_prob, pn_feat = pn_prob_feat[:,0], pn_prob_feat[:,1:]
        pos_prob_feat = load_tensor[:,1:,:]
        pos_prob, pos_feat = pos_prob_feat[:,:,0], pos_prob_feat[:,:,1:]
        # step1: 按pn_prob从大到小排序
        sorted_idx = torch.argsort(pn_prob, descending=True)  # (L,)
        pn_feat_sorted = pn_feat[sorted_idx]                  # (L, dim)

        # step2: 取pos_prob前k个最大值对应的pos_feat
        top_idx = torch.topk(pos_prob, k=3, dim=1).indices    # (L, 3)
        top_pos_feat = torch.gather(pos_feat, 1, top_idx.unsqueeze(-1).expand(-1, -1, pos_feat.size(-1)))  # (L, 3, dim)
        pos_feat_sum = top_pos_feat.mean(dim=1)   # (L, dim)

        feat_concat = torch.cat([pn_feat_sorted, pos_feat_sum], dim=1)   # (L, dim*2)
        slide_tensor = feat_concat[sorted_idx[:self.patch_nums]]   # (topk, dim*2)
        
        return slide_tensor

    def format_slide_tensor_nopn(self, feat_path):
        '''no pos/neg prob and feat: ml_decoder'''
        load_tensor = torch.load(feat_path)    # (L, dim)
        pos_prob, pos_feat = load_tensor[:,:,0], load_tensor[:,:,1:]
        # 取每个patch的最大pos_prob
        max_prob, _ = torch.max(pos_prob, dim=1)   # (L,)
        # 按最大prob从大到小排序，得到索引
        sorted_idx = torch.argsort(max_prob, descending=True)   # (L,)

        # 取pos_prob前k个最大值对应的pos_feat
        top_idx = torch.topk(pos_prob, k=3, dim=1).indices    # (L, 3)
        top_pos_feat = torch.gather(pos_feat, 1, top_idx.unsqueeze(-1).expand(-1, -1, pos_feat.size(-1)))  # (L, 3, dim)
        pos_feat_sum = top_pos_feat.sum(dim=1)   # (L, dim)

        slide_tensor = pos_feat_sum[sorted_idx[:self.patch_nums]]   # (topk, dim)

        return slide_tensor

    def format_slide_tensor_simple(self, feat_path):
        ''''''
        load_tensor = torch.load(feat_path)    # (L, dim)
        slide_tensor = load_tensor[:self.patch_nums]   # (topk, dim)
        return slide_tensor

