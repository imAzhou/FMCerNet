from PIL import Image
from torch.utils.data import Dataset
from mmcv.transforms import Compose
import json
import torch
from tqdm import tqdm
from cerwsi.utils import is_main_process

# 自定义数据集类
class ClsDataset(Dataset):
    def __init__(self, data_cfg, annojson_path, transform):
        """
        Args:
            img_dir (str): img dir
        """
        self.img_dir = data_cfg.img_dir
        self.transform = Compose(transform)
        self.annojson_path = annojson_path
        with open(self.annojson_path, 'r') as f:
            self.patch_COCOinfo = json.load(f)
        
        self.format_COCO2mmpretrain()
    
    def format_COCO2mmpretrain(self):
        self.imginfo_list = []
        pbar = self.patch_COCOinfo['images']
        if is_main_process():
            pbar = tqdm(self.patch_COCOinfo['images'], ncols=80)
        
        for imginfo in pbar:
            imginfo['img_path'] = f'{self.img_dir}/{imginfo["file_name"]}'
            self.imginfo_list.append(imginfo)

    def __len__(self):
        return len(self.patch_COCOinfo['images'])

    def __getitem__(self, idx):
        output = self.transform(self.imginfo_list[idx])
        output['data_samples'].diagnose = self.imginfo_list[idx]['diagnose']
        output['data_samples'].extra_info = self.imginfo_list[idx]['extra_info']
        return output
