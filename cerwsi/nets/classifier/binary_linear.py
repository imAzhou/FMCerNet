import torch
from torch import nn
import torch.nn.functional as F
from .meta_classifier import MetaClassifier
from cerwsi.utils import build_evaluator, BinaryMetric


class BinaryLinear(MetaClassifier):
    def __init__(self, args):
        input_embed_dim = args.backbone_cfg['backbone_output_dim'][-1]
        evaluator = build_evaluator([BinaryMetric(args.logger_name, thr = args.positive_thr)])
        super(BinaryLinear, self).__init__(evaluator, **args)

        self.backbone_type = args.backbone_type
        self.cls_linear_head = nn.Linear(input_embed_dim, 1)

    def calc_logits(self, inputs):
        if self.backbone_type == 'resnet':
            feat = inputs[-1]
            feat = feat.mean(dim=[2, 3])
        elif self.backbone_type == 'sam2':
            feat = inputs['vision_features']
            feat = feat.mean(dim=[2, 3])
        elif self.backbone_type in ['dinov2', 'uni']:
            feat = inputs[:,0,:]
        elif self.backbone_type == 'smartccs':
            feat = inputs['x_norm_clstoken']
        
        pred_img_logits = self.cls_linear_head(feat)  # (bs, 1)
        return pred_img_logits
    
    def calc_loss(self,inputs, databatch):
        img_pn_logit = self.calc_logits(inputs)
        image_labels = torch.tensor([int(len(item.gt_label)>0) for item in databatch['data_samples']])
        img_gt = image_labels.to(self.device).unsqueeze(-1).float()
        pn_loss = F.binary_cross_entropy_with_logits(img_pn_logit, img_gt, reduction='mean')
        loss_dict = {
            'pn_loss': pn_loss.item(),
        }
        return pn_loss,loss_dict

    def set_pred(self,inputs, databatch):
        img_pn_logit = self.calc_logits(inputs) # (bs, 1)
        img_probs = torch.sigmoid(img_pn_logit).squeeze(-1)   # (bs, )
        data_sampels = []
        for item, pn_p in zip(databatch['data_samples'], img_probs):
            item.img_prob = pn_p
            data_sampels.append(item)
        return data_sampels
