import torch
from torch.utils.data import Dataset
import json
from mmcv.transforms import Compose
from collections import defaultdict
from mmdet.models.utils import mask2ndarray

from tqdm import tqdm

# 自定义数据集类
class InstanceDataset(Dataset):
    def __init__(self, data_cfg, annojson_path, transform):
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
        
        self.format_COCO2mmdet()

    def format_COCO2mmdet(self):
        print('Start to format annotaion in mmdet style.')
        annoInimg = defaultdict(list)
        for annoinfo in self.patch_COCOinfo['annotations']:
            x,y,w,h = annoinfo['bbox']
            annoInimg[annoinfo['image_id']].append({
                'bbox': [x, y, x+w, y+h],
                'bbox_label': annoinfo['category_id'],
                'mask': annoinfo['segmentation'],
                'inst_id': annoinfo['inst_id'],
                'ignore_flag': False     # False means retain 
            })

        self.imginfo_list = []
        for imginfo in tqdm(self.patch_COCOinfo['images'], ncols=80):
            imginfo['img_id'] = imginfo['id']
            imginfo['img_path'] = f'{self.img_dir}/{imginfo["file_name"]}'
            imginfo['instances'] = annoInimg[imginfo['id']]
            self.imginfo_list.append(imginfo)
        print('Done to format annotaion in mmdet style.')

    def __len__(self):
        return len(self.patch_COCOinfo['images'])

    def __getitem__(self, idx):
        output = self.transform(self.imginfo_list[idx])

        output['data_samples'].diagnose = self.imginfo_list[idx]['diagnose']
        output['data_samples'].prefix = self.imginfo_list[idx]['prefix']
        output['data_samples'].gt_instances.masks = torch.as_tensor(
            mask2ndarray(output['data_samples'].gt_instances.masks))
        return output
