import random
import torch
import torch.nn as nn
import math
import random
import torch.nn.functional as F
from ..meta_classifier import MetaClassifier
from cerwsi.utils import build_evaluator,ImgODMetric,ImgODCOCOMetric,BinaryMetric
from .binary_cls_branch import BinaryClsBranch
from .instance_branch import Instance_branch


def get_featcfg(args):
    if args.backbone_type == 'sam2':
        backbone_stride = 16
        input_featsize = args.input_size // backbone_stride
        input_feat = [
            (256, input_featsize,input_featsize),    
            (256, input_featsize*2,input_featsize*2),
            (256, input_featsize*4,input_featsize*4),
        ]
        upscaled_feat = [
            (256//4, input_featsize*2,input_featsize*2),
            (256//8, input_featsize*4,input_featsize*4),
        ]
    elif args.backbone_type == 'convnext':
        backbone_stride = 32
        input_featsize = args.input_size // backbone_stride
        input_feat = [
            (1536, input_featsize,input_featsize),    
            (768, input_featsize*2,input_featsize*2),
            (384, input_featsize*4,input_featsize*4),
        ]
        upscaled_feat = [
            (768, input_featsize*2,input_featsize*2),
            (384, input_featsize*4,input_featsize*4),
        ]
    
    featcfg = {
        'input_feat': input_feat,
        'upscaled_feat': upscaled_feat
    }
    return featcfg

class WSCerPartial(MetaClassifier):
    def __init__(self, args):

        save_result_dir = getattr(args, 'save_result_dir', None)
        evaluator = build_evaluator([ImgODCOCOMetric(
            args.logger_name,save_result_dir,
            args.val_evaluator,args.classes
        )])
        # evaluator = build_evaluator([BinaryMetric(args.logger_name, 
        #                                           thr = args.positive_thr,
        #                                           save_result_dir = save_result_dir,)])
        super(WSCerPartial, self).__init__(evaluator, **args)

        featcfg = get_featcfg(args)
        self.binary_cls_branch = BinaryClsBranch(featcfg['input_feat'][0][0])
        self.instance_branch = Instance_branch(
            img_input_size = args.input_size,
            num_classes = args.num_classes,
            num_instance_queries = args.num_instance_queries,
            pretrain_ckpt = args.instance_ckpt,
            featcfg = featcfg
        )

    def filter4inst(self, dict_inputs: dict, databatch):
        posIndx = [idx for idx,item in enumerate(databatch['data_samples']) if item.diagnose==1]
        negIndx = [idx for idx,item in enumerate(databatch['data_samples']) if item.diagnose==0]
        sample_cnt = max(2, len(posIndx)//4)
        sample_neg = random.sample(negIndx, sample_cnt)
        choice_idx = [*posIndx, *sample_neg]
        
        filter_dict_inputs = {
            'vision_features': dict_inputs['vision_features'][posIndx],
            # 'vision_pos_enc': [feat[posIndx] for feat in dict_inputs['vision_pos_enc']],
            'backbone_fpn': [feat[posIndx] for feat in dict_inputs['backbone_fpn']],
        }
        filter_databatch = {
            'inputs': databatch['inputs'][choice_idx],
            'data_samples': [databatch['data_samples'][i] for i in choice_idx],
            'image_labels': databatch['image_labels'][choice_idx]
        }

        return filter_dict_inputs,filter_databatch

    def calc_loss(self, dict_inputs: dict, databatch):
        '''
        dict_inputs: dict, 
            vision_features: Tensor, (bs, c, h, w)
            vision_pos_enc: List[Tensor]: [bs, c, h1,w1]...
            backbone_fpn: List[Tensor]: [bs, c, h1,w1]...
        '''
        img_logits,inter_var = self.binary_cls_branch(dict_inputs['vision_features'])
        # patch_probs = self.binary_cls_branch.patch_probs(inter_var, scale=4)
        binary_loss_fn = nn.BCEWithLogitsLoss()
        img_gt = databatch['image_labels'].unsqueeze(1).float()
        img_loss = binary_loss_fn(img_logits, img_gt)

        loss = img_loss
        loss_dict = {'img_loss': img_loss.item()}

        inst_dict_inputs,inst_databatch = self.filter4inst(dict_inputs, databatch)
        # inst_dict_inputs,inst_databatch = dict_inputs, databatch
        instance_loss_dict = self.instance_branch.loss(inst_dict_inputs,inst_databatch, None)
        loss = 0.
        loss_dict = {}
        for key,value in instance_loss_dict.items():
            loss += value
            loss_dict[key] = value.item()

        return loss, loss_dict
    
    def set_pred(self, dict_inputs, databatch):
        img_logits,inter_var = self.binary_cls_branch(dict_inputs['vision_features'])
        databatch['img_probs'] = torch.sigmoid(img_logits)
        
        # patch_probs = self.binary_cls_branch.patch_probs(inter_var, scale=4)
        # databatch['patch_probs'] = patch_probs
        databatch['pred_bbox'] = self.instance_branch.predict(dict_inputs, databatch, None)
        databatch['img_probs'] = torch.Tensor([int(len(predbbox)>0) for predbbox in databatch['pred_bbox']])
        return databatch
