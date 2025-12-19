import json
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from mmcv.transforms import Compose
import os
from mmpretrain.structures import DataSample

# 自定义数据集类
class AttriDataset(Dataset):
    def __init__(self, cfg, jsonfile, transform):
        with open(jsonfile, 'r', encoding='utf-8') as f:
            self.data_list = json.load(f)
        
        self.img_dir = cfg.img_dir
        self.transform = Compose(transform)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        cell_info = self.data_list[idx]
        output = self.transform(dict(
            img_path = f'{self.img_dir}/{cell_info["filename"]}'
        ))
        output['data_samples'].attr_v = cell_info["attr_v"]
        output['data_samples'].sub_class = cell_info["sub_class"]

        return output
    
    

if __name__ == "__main__":
    with open('data_resource/cell_attri/train_cellinst.json', 'r', encoding='utf-8') as f:
        json_data = json.load(f)

        print()