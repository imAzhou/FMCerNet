import torch
from torch.utils.data import Dataset
import json
from mmcv.transforms import Compose
from collections import defaultdict
from mmdet.models.utils import mask2ndarray
import numpy as np
from tqdm import tqdm
from pycocotools import mask as mask_utils
from mmdet.structures.bbox import bbox_mapping

# 自定义数据集类
class InstanceDataset(Dataset):
    def __init__(self, data_cfg, annojson_path, transform):
        """
        Args:
            img_dir (str): img dir
        """
        self.img_dir = data_cfg.img_dir
        self.num_classes = data_cfg.num_classes
        self.classes = data_cfg.classes
        self.load_proposal = data_cfg.get('load_proposal', False)
        if self.load_proposal:
            self.proposal_dir = f'{data_cfg.data_root}/sam2Infer'
        
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
                # 'inst_id': annoinfo['inst_id'],
                'ignore_flag': False     # False means retain 
            })

        self.imginfo_list = []
        for imginfo in tqdm(self.patch_COCOinfo['images'], ncols=80):
            imginfo['img_id'] = imginfo['id']
            imginfo['img_path'] = f'{self.img_dir}/{imginfo["file_name"]}'
            imginfo['instances'] = annoInimg[imginfo['id']]
            if self.load_proposal:
                imginfo['sam2proposals'] = self.load_proposal_fn(imginfo["file_name"])
            
            self.imginfo_list.append(imginfo)
        print('Done to format annotaion in mmdet style.')

    def load_proposal_fn(self, filename):
        jsonfilename = filename.replace('.png', '.json')
        with open(f"{self.proposal_dir}/{jsonfilename}", 'r', encoding='utf-8') as f:
            self.proposals = json.load(f)
        bboxes,scores,masks = [],[],[]
        for proposalinfo in self.proposals:
            x1, y1, w, h = proposalinfo['bbox']
            bboxes.append([x1, y1, x1+w, y1+h])
            scores.append(proposalinfo['stability_score'])
            # masks.append(mask_utils.decode(proposalinfo['segmentation']))
        return {
            'bboxes': np.array(bboxes), 
            'scores': np.array(scores),
            # 'masks': np.array(masks)
        }

    def __len__(self):
        return len(self.patch_COCOinfo['images'])

    def __getitem__(self, idx):
        output = self.transform(self.imginfo_list[idx])

        output['data_samples'].diagnose = self.imginfo_list[idx]['diagnose']
        output['data_samples'].extra_info = self.imginfo_list[idx]['extra_info']
        output['data_samples'].gt_instances.masks = torch.as_tensor(
            mask2ndarray(output['data_samples'].gt_instances.masks))
        
        if self.load_proposal:
            sf = output['data_samples'].scale_factor
            sam2proposals = self.imginfo_list[idx]['sam2proposals']
            proposal_bbox = bbox_mapping(
                torch.Tensor(sam2proposals['bboxes']),
                output['data_samples'].img_shape,
                (sf[0], sf[1], sf[0], sf[1]),
                flip=output['data_samples'].get('flip', False),
                flip_direction=output['data_samples'].get('flip_direction', None)
            )
            output['data_samples'].sam2proposal = {
                'bboxes': proposal_bbox,
                'scores': torch.Tensor(sam2proposals['scores'])
            }
        return output
