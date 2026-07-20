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
        self.C_in = cfg.get('dataset_in_dim', cfg.in_dim)
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
            if self.format_type == 'pn_only':
                slide_tensor = self.format_slide_tensor_pn_only(feat_path)
            elif self.format_type == 'pn_posprob':
                slide_tensor = self.format_slide_tensor_pn_posprob(feat_path)
            elif self.format_type == 'pos_only_top1':
                slide_tensor = self.format_slide_tensor_pos_top1(feat_path)
            elif self.format_type == 'all_prob_weighted':
                slide_tensor = self.format_slide_tensor_all_prob_weighted(feat_path)
            elif self.format_type == 'raw_pn_pos_tokens':
                slide_tensor = self.format_slide_tensor_raw_tokens(feat_path)
            else:
                raise ValueError(f'Invalid format_type: {self.format_type}')
        else:
            attn_mask[:] = 1

        L, dim = slide_tensor.shape
        if os.path.exists(feat_path):
            attn_mask[:min(L, self.patch_nums)] = 1
        if L > self.patch_nums:
            slide_tensor = slide_tensor[:self.patch_nums, :]
        elif L < self.patch_nums:
            pad_size = (0, 0, 0, self.patch_nums - L)  
            # pad_size 含义 (dim2_pad_left, dim2_pad_right, dim1_pad_left, dim1_pad_right)
            # 这里在第 0 维（序列长度）末尾补 (target_len - L) 行
            slide_tensor = F.pad(slide_tensor, pad_size, value=0)

        # slide_tensor = torch.rand(self.patch_nums, 517)
        slide_clsname = slide_info['slide_clsname']
        if self.cls_map is not None:
            slide_clsname = self.cls_map[slide_info['slide_clsname']]
        slide_label = self.classes.index(slide_clsname)
        data_samples = DataSample()
        data_samples.slide_label = slide_label
        data_samples.slide_info = slide_info
        data_samples.attn_mask = attn_mask
        return {
            'inputs': slide_tensor,
            'data_samples': data_samples
        }
    
    def format_slide_tensor_pn_only(self, feat_path):
        load_tensor = torch.load(feat_path)    # (L, 6, 513)
        pn_prob_feat = load_tensor[:, 0, :]
        pn_prob, pn_feat = pn_prob_feat[:, 0], pn_prob_feat[:, 1:]
        sorted_idx = torch.argsort(pn_prob, descending=True)
        return pn_feat[sorted_idx[:self.patch_nums]]

    def format_slide_tensor_pn_posprob(self, feat_path):
        load_tensor = torch.load(feat_path)    # (L, 6, 513)
        pn_prob_feat = load_tensor[:, 0, :]
        pn_prob, pn_feat = pn_prob_feat[:, 0], pn_prob_feat[:, 1:]
        pos_prob = load_tensor[:, 1:, 0]
        feat_concat = torch.cat([pn_feat, pos_prob], dim=1)
        sorted_idx = torch.argsort(pn_prob, descending=True)
        return feat_concat[sorted_idx[:self.patch_nums]]

    def format_slide_tensor_pos_top1(self, feat_path):
        load_tensor = torch.load(feat_path)    # (L, 6, 513)
        pn_prob_feat = load_tensor[:, 0, :]
        pn_prob, pn_feat = pn_prob_feat[:, 0], pn_prob_feat[:, 1:]
        pos_prob_feat = load_tensor[:, 1:, :]
        pos_prob, pos_feat = pos_prob_feat[:, :, 0], pos_prob_feat[:, :, 1:]
        sorted_idx = torch.argsort(pn_prob, descending=True)
        top_idx = torch.argmax(pos_prob, dim=1)
        top_pos_feat = pos_feat[torch.arange(pos_feat.size(0)), top_idx]
        feat_concat = torch.cat([pn_feat, top_pos_feat], dim=1)
        return feat_concat[sorted_idx[:self.patch_nums]]

    def format_slide_tensor_all_prob_weighted(self, feat_path):
        load_tensor = torch.load(feat_path)    # (L, 6, 513)
        pn_prob_feat = load_tensor[:, 0, :]
        pn_prob, pn_feat = pn_prob_feat[:, 0], pn_prob_feat[:, 1:]
        pos_prob_feat = load_tensor[:, 1:, :]
        pos_prob, pos_feat = pos_prob_feat[:, :, 0], pos_prob_feat[:, :, 1:]
        pos_prob_sum = pos_prob.sum(dim=1, keepdim=True)
        if torch.any(pos_prob_sum <= 0):
            raise ValueError(f'Non-positive pos_prob sum in {feat_path}.')
        pos_weight = pos_prob / pos_prob_sum
        pos_feat_weighted = (pos_weight.unsqueeze(-1) * pos_feat).sum(dim=1)
        feat_concat = torch.cat([pn_feat, pos_feat_weighted], dim=1)
        sorted_idx = torch.argsort(pn_prob, descending=True)
        return feat_concat[sorted_idx[:self.patch_nums]]

    def format_slide_tensor_raw_tokens(self, feat_path):
        load_tensor = torch.load(feat_path)    # (L, 6, 513)
        pn_prob = load_tensor[:, 0, 0]
        sorted_idx = torch.argsort(pn_prob, descending=True)
        sorted_tensor = load_tensor[sorted_idx[:self.patch_nums]]
        return sorted_tensor.flatten(start_dim=1)
