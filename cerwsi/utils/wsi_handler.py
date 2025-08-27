from math import ceil
import copy
import torch
import cv2
from PIL import Image
import numpy as np
import random
from mmpretrain.structures import DataSample

from .KFBreader.kfbreader import KFBSlide,kfbslide_get_associated_image_names,kfbslide_read_associated_image

class WSIHandler:
    def __init__(self, kfb_path, crop_ws,
                 level=0, 
                 safe_margin=100, 
                 certain_thr=0.7,
                 positive_thr=0.7,
                 ):
        self.slide = KFBSlide(kfb_path)
        self.level = level
        self.safe_margin = safe_margin
        self.crop_ws = crop_ws
        self.certain_thr = certain_thr
        self.positive_thr = positive_thr

    def save_thumbnail(self, savepath):
        smallest_level = len(self.slide.level_downsamples)-1
        width, height = self.slide.level_dimensions[smallest_level]
        location, level, size = (0, 0), smallest_level, (width, height)
        read_result = Image.fromarray(self.slide.read_region(location, level, size))
        read_result.save(savepath)

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
    
    def read_PILimg(self, point_xy):
        '''
        return RGB PIL.Image imgdata, value in [0,1]
        '''
        x,y = point_xy
        location, level, size = (x, y), self.level, (self.crop_ws, self.crop_ws)
        read_result = copy.deepcopy(Image.fromarray(self.slide.read_region(location, level, size)))
        return read_result

    def inference_valid_batch(self, valid_m, valid_datapool):
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


    def inference_batch_pn(self, pn_m, pn_datapool):
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
            outputs = pn_m(data_batch, 'val')
        for inputInfo, datasample in zip(pn_datapool, outputs):
            pred_clsid = int(datasample.img_prob > self.positive_thr)
            inputInfo['img_prob'] = datasample.img_prob.item()
            inputInfo['pred_label'] = pred_clsid
            inputInfo['img_token'] = datasample.img_token
        return pn_datapool

    def format_logstr(self, slide_patchlist):
        return "total:{}, invalid:{}, uncertain:{}, valid:{}, neg:{}, pos:{}".format(
            len(slide_patchlist),
            sum(p['valid_flag']==0 for p in slide_patchlist),
            sum(p['valid_flag']==1 for p in slide_patchlist),
            sum(p['valid_flag']==2 for p in slide_patchlist),
            sum(p['valid_flag']==2 and p['pred_label']==0 for p in slide_patchlist),
            sum(p['valid_flag']==2 and p['pred_label']==1 for p in slide_patchlist)
        )
    