import torch
from torch.utils.data import Dataset
import pandas as pd

# 自定义数据集类
class SlideDataset(Dataset):
    def __init__(self, cfg, csvfile):
        df_data = pd.read_csv(csvfile)
        df_data = df_data.drop_duplicates(subset=["patientId"])   # 按 patientId 去重
        self.data_list = df_data.to_dict(orient="records")  # 每一行 -> dict
        self.feat_dir = cfg.feat_dir
        self.classes = cfg.classes
        self.cls_map = cfg.cls_map

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        slide_info = self.data_list[idx]
        patientId = slide_info['patientId']
        # slide_tensor = torch.load(f'{self.feat_dir}/{patientId}.pt')
        slide_tensor = torch.rand(100, 517)
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
