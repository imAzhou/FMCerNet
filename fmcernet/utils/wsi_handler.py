from math import ceil
import copy
import torch
import cv2
from PIL import Image
from torchvision import transforms
import numpy as np
import random
from mmpretrain.structures import DataSample
from mmdet.structures import DetDataSample
from .tools import is_bbox_inside
from .KFBreader.kfbreader import KFBSlide



class WSIHandler:
    def __init__(self, source_path, crop_ws,
                 level=0, 
                 safe_margin=100, 
                 certain_thr=0.7,
                 positive_thr=0.7,  # patch pos thr
                 bbox_score_thr=0.2,
                 positive_class=[],
                 ):
        self.slide = KFBSlide(source_path)
        self.level = level
        self.safe_margin = safe_margin
        self.crop_ws = crop_ws
        self.certain_thr = certain_thr
        self.positive_thr = positive_thr
        self.bbox_score_thr = bbox_score_thr
        self.positive_class = positive_class

    def save_thumbnail(self, savepath):
        smallest_level = len(self.slide.level_downsamples)-1
        width, height = self.slide.level_dimensions[smallest_level]
        location, level, size = (0, 0), smallest_level, (width, height)
        read_result = Image.fromarray(self.slide.read_region(location, level, size))
        read_result.save(savepath)
        return read_result

    def init_patchlist(self, init_dict):
        width, height = self.slide.level_dimensions[self.level]
        width -= self.safe_margin
        height -= self.safe_margin
        iw, ih = ceil(width/self.crop_ws), ceil(height/self.crop_ws)
        r2 = (int(max(iw, ih)*1.1)//2)**2
        cix, ciy = iw // 2, ih // 2
        slide_patchlist = []
        for j, y in enumerate(range(0, height, self.crop_ws)):
            for i, x in enumerate(range(0, width, self.crop_ws)):
                if (i-cix)**2 + (j-ciy)**2 > r2:
                    continue
                _init_dict = copy.deepcopy(init_dict)
                _init_dict['xy'] = (x,y)
                _init_dict['coords'] = [x, y, x+self.crop_ws, y+self.crop_ws]
                slide_patchlist.append(_init_dict)
        return slide_patchlist
    
    def read_cv2img(self, point_xy=None, random_cut=False):
        '''
        return BGR numpy imgdata, value in [0,255]
        '''
        if random_cut:
            max_x, max_y = self.slide.level_dimensions[self.level]
            max_x, max_y = max_x-self.safe_margin, max_y-self.safe_margin
            x1,y1 = random.randint(self.safe_margin, max_x-self.crop_ws),random.randint(self.safe_margin, max_y-self.crop_ws)
            point_xy = (x1,y1)

        read_result = self.read_PILimg(point_xy)
        img_input = cv2.cvtColor(np.array(read_result), cv2.COLOR_RGB2BGR)
        x1,y1 = point_xy
        coords = [x1, y1, x1+self.crop_ws, y1+self.crop_ws]
        return img_input,coords
    
    def read_PILimg(self, point_xy, bboxwh=None):
        '''
        Args:
            point_xy: left top coord (x, y) 
            bboxwh: region crop size (width, height) 
        return RGB PIL.Image imgdata, value in [0,1]
        '''
        x,y = point_xy
        if bboxwh == None:
            bboxwh = (self.crop_ws, self.crop_ws)
        location, level, size = (x, y), self.level, bboxwh
        read_result = copy.deepcopy(Image.fromarray(self.slide.read_region(location, level, size)))
        return read_result

    def infer_valid_fn(self, valid_m, valid_datapool):
        '''
        Need attribute: image
        Add attribute: valid_flag, valid_prob
        '''
        data_batch = dict(inputs=[], data_samples=[])
        for item in valid_datapool:
            img_input = torch.as_tensor(cv2.resize(item['image'], (224,224)))
            data_batch['inputs'].append(img_input.permute(2,0,1))    # (bs, 3, h, w)
            data_batch['data_samples'].append(DataSample())

        data_batch['inputs'] = torch.stack(data_batch['inputs'], dim=0)
        with torch.no_grad():
            outputs = valid_m.val_step(data_batch)
        for input_patch,pred_output in zip(valid_datapool,outputs):
            valid_flag = 1
            if max(pred_output.pred_score) > self.certain_thr:
                valid_flag = 2 if pred_output.pred_label == 1 else 0
            input_patch['valid_flag'] = valid_flag
            input_patch['valid_prob'] = pred_output.pred_score[1].item()

    def infer_pn_fn(self, pn_m, pn_datapool):
        '''
        Need attribute: image
        Add attribute: img_prob, pred_label, img_token
        '''
        inputsize = pn_m.img_size
        data_batch = dict(inputs=[], data_samples=[])
        for item in pn_datapool:
            img_input = torch.as_tensor(cv2.resize(item['image'], (inputsize,inputsize)))
            data_batch['inputs'].append(img_input.permute(2,0,1))    # (bs, 3, h, w)
            data_batch['data_samples'].append(DataSample())

        data_batch['inputs'] = torch.stack(data_batch['inputs'], dim=0)
        with torch.no_grad():
            outputs = pn_m(data_batch, 'predict')
        for inputInfo, datasample in zip(pn_datapool, outputs):
            pred_clsid = int(datasample.img_prob > self.positive_thr)
            inputInfo['img_prob'] = datasample.img_prob.item()
            inputInfo['pred_label'] = pred_clsid
            if 'img_token' in inputInfo:
                inputInfo['img_token'] = datasample.img_token.detach().cpu()
            if 'img_attnmap' in inputInfo:
                inputInfo['img_attnmap'] = datasample.attn
        return pn_datapool

    def infer_pn_batch_fn(self, pn_m, datadict_list, batch_size):
        inputsize = pn_m.img_size
        datapool = []
        valid_list = [item for item in datadict_list if item['valid_flag'] != 0]
        for didx, item in enumerate(valid_list):
            img_input = torch.as_tensor(cv2.resize(item['image'], (inputsize,inputsize)))
            datapool.append(img_input.permute(2,0,1))
            
            if len(datapool) % batch_size == 0 or didx == len(valid_list)-1:
                with torch.no_grad():
                    outputs = pn_m({
                        'inputs': torch.stack(datapool, dim=0),
                        'data_samples': [DataSample() for _ in range(len(datapool))]
                    }, 'predict')
                start_idx = didx-len(datapool)+1
                for oidx,datasample in enumerate(outputs):
                    tgt_item = valid_list[start_idx + oidx]
                    pred_clsid = int(datasample.img_prob > self.positive_thr)
                    tgt_item['img_prob'] = datasample.img_prob.item()
                    tgt_item['pred_label'] = pred_clsid
                    if 'img_token' in tgt_item:
                        tgt_item['img_token'] = datasample.img_token.detach().cpu()
                    if 'img_attnmap' in tgt_item:
                        tgt_item['img_attnmap'] = datasample.attn
                    del tgt_item['image']
                datapool = []
                torch.cuda.empty_cache()
        # clear invalid img
        for item in datadict_list:
            if item['valid_flag'] == 0:
                del item['image']
        torch.cuda.empty_cache()

    def infer_celldetector_fn(self, detector, pos_datapool):
        '''
        Need attribute: image
        Add attribute: pred_bboxes
        '''
        inputsize = detector.img_size
        imgw,imgh = self.crop_ws,self.crop_ws
        transform = transforms.Compose([
            transforms.Resize(inputsize),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        img_inputs = [transform(
            Image.fromarray(cv2.cvtColor(item['image'],cv2.COLOR_BGR2RGB))
        ) for item in pos_datapool]
        images_tensor = torch.stack(img_inputs, dim=0).to(detector.device)
        data_batch = dict(
            inputs = images_tensor,
            data_samples = [
                DetDataSample(
                    metainfo = {
                        'img_shape':(inputsize,inputsize),
                        'ori_shape':(imgw,imgh),
                        'scale_factor': (inputsize/imgh, inputsize/imgh)
                    },
                    batch_input_shape = (inputsize,inputsize),
                ) for i in range(len(pos_datapool))]
        )
        with torch.no_grad():
            outputs = detector(data_batch['inputs'], data_batch['data_samples'], mode="predict")
        for input_patch,pred_output in zip(pos_datapool,outputs):
            predresult = pred_output.pred_instances
            input_patch['pred_bboxes'] = self.detector_postprocess(predresult,input_patch['coords'])
        return pos_datapool
    
    def detector_postprocess(self, predresult, pcoords):
        pred_bboxes,pred_scores,pred_labels = predresult.bboxes.cpu(), predresult.scores.cpu(), predresult.labels.cpu()
        px1, py1, px2, py2 = pcoords
        new_bboxes = []
        filtered_bboxes,filtered_scores,filtered_labels = [],[],[]
        for bbox,score,label in zip(pred_bboxes,pred_scores,pred_labels):
            if score >= self.bbox_score_thr:
                filtered_bboxes.append(bbox.tolist()) # x1,y1,x2,y2
                filtered_scores.append(score.item())
                filtered_labels.append(label.item())
        if len(filtered_bboxes) == 0:
            return new_bboxes
        
        bboxes,labels = np.array(filtered_bboxes),np.array(filtered_labels)
        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        sorted_indices = np.argsort(-areas)  # 从大到小排序

        n = len(bboxes)
        used = np.zeros(n, dtype=bool)
        for i in sorted_indices:
            if used[i]:
                continue
            group_indices = [i]
            for j in sorted_indices:
                if i == j or used[j]:
                    continue
                if is_bbox_inside(bboxes[j], bboxes[i], tolerance=5):
                    group_indices.append(j)
                    used[j] = True

            # 当前 group 最外层是 i，label 取 group 内最大值
            max_label = labels[group_indices].max()
            bx1, by1, bx2, by2 = bboxes[i].tolist()
            bx1, by1, bx2, by2 = bx1+px1, by1+py1, bx2+px1, by2+py1
            new_bboxes.append({
                'parent_coord': pcoords,   # bbox 所在 patch 块坐标（在 LEVEL=0 上的绝对坐标）
                'coord': [bx1, by1, bx2, by2],  # bbox 在 LEVEL=0 上的绝对坐标
                'score': filtered_scores[i],     # bbox 阳性类别置信度
                'label': int(max_label),
                'clsname': self.positive_class[int(max_label)]    # bbox 阳性类别名称
            })
            used[i] = True

        return new_bboxes

    def format_logstr(self, slide_patchlist):
        if 'pred_label' in slide_patchlist[0]:
            return "total:{}, invalid:{}, uncertain:{}, valid:{}, neg:{}, pos:{}".format(
                len(slide_patchlist),
                sum(p['valid_flag']==0 for p in slide_patchlist),
                sum(p['valid_flag']==1 for p in slide_patchlist),
                sum(p['valid_flag']==2 for p in slide_patchlist),
                sum(p['valid_flag']!=0 and p['pred_label']==0 for p in slide_patchlist),
                sum(p['valid_flag']!=0 and p['pred_label']==1 for p in slide_patchlist)
            )
        else:
            return "total:{}, invalid:{}, uncertain:{}, valid:{}".format(
                len(slide_patchlist),
                sum(p['valid_flag']==0 for p in slide_patchlist),
                sum(p['valid_flag']==1 for p in slide_patchlist),
                sum(p['valid_flag']==2 for p in slide_patchlist),
            )
