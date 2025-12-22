import torch
from torch import nn
from .meta_classifier import MetaClassifier
from cerwsi.utils import build_evaluator, MultiClsMetric


class MCLinear(MetaClassifier):
    def __init__(self, args):
        evaluator = build_evaluator([MultiClsMetric(
            num_classes = args.num_classes,
            classes = args.classes,
            logger_name = args.logger_name,
        )])
        super(MCLinear, self).__init__(evaluator, args)
        input_embed_dim = args.backbone_cfg['backbone_output_dim'][-1]
        self.num_classes = args.num_classes
        self.backbone_type = args.backbone_type
        self.cls_linear_heads = nn.Linear(input_embed_dim, self.num_classes)
    
    def calc_logits(self, inputs):
        cls_token = self.get_cls_token(inputs)
        pred_logits = self.cls_linear_heads(cls_token)  # （bs, num_cls）
        return pred_logits
    
    def calc_loss(self, inputs, databatch):
        loss_fn = nn.CrossEntropyLoss()
        pred_logits = self.calc_logits(inputs)
        gt_labels = torch.cat([item.gt_label for item in databatch['data_samples']]).to(self.device)
        loss = loss_fn(pred_logits, gt_labels)
        loss_dict = {
            'loss': loss.item(),
        }
        return loss,loss_dict

    def set_pred(self,inputs, databatch):
        pred_logits = self.calc_logits(inputs) # (bs, num_classes)
        pred_probs = torch.softmax(pred_logits, dim=-1) # (bs, n_cls)
        _,pred_labels = torch.max(pred_probs, dim=-1)
        data_sampels = []
        for item, pred_prob, pred_label in zip(databatch['data_samples'], pred_probs, pred_labels):
            item.pred_prob = pred_prob
            item.pred_label = pred_label
            data_sampels.append(item)

        return data_sampels
