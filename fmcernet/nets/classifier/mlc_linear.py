import torch
from torch import nn
from .meta_classifier import MetaClassifier
from fmcernet.utils import build_evaluator, ExtendMultiLabelMetric, build_loss


class MLCLinear(MetaClassifier):
    def __init__(self, args):
        evaluator = build_evaluator([ExtendMultiLabelMetric(
            thr = args.positive_thr,
            num_classes = args.num_classes,
            logger_name = args.logger_name,
            with_binary = False
        )])
        super(MLCLinear, self).__init__(evaluator, args)
        input_embed_dim = args.backbone_cfg['backbone_output_dim'][-1]
        self.num_classes = args.num_classes
        self.backbone_type = args.backbone_type
        self.cls_linear_heads = nn.ModuleList()
        for i in range(self.num_classes):
            self.cls_linear_heads.append(nn.Linear(input_embed_dim, 1))
        self.loss_fn = build_loss(args.loss_cfg)
    
    def calc_logits(self, inputs):
        cls_token = self.get_cls_token(inputs)
        pred_pos_logits = []
        for i in range(self.num_classes):
            pred_pos_logits.append(self.cls_linear_heads[i](cls_token))  # [(bs, 1),]
        pred_pos_logits = torch.cat(pred_pos_logits, dim=-1)
        return pred_pos_logits
    
    
    def calc_loss(self, inputs, databatch):
        pred_logits = self.calc_logits(inputs)
        binary_matrix = self.get_mlc_labels(databatch)
        loss = self.loss_fn(pred_logits, binary_matrix)
        loss_dict = {
            'pos_loss': loss.item(),
        }
        return loss,loss_dict

    def set_pred(self,inputs, databatch):
        positive_logits = self.calc_logits(inputs) # (bs, num_classes)
        pos_probs = torch.sigmoid(positive_logits) # (bs, n_cls)
        data_sampels = []
        for item, pos_p in zip(databatch['data_samples'], pos_probs):
            item.pos_prob = pos_p
            data_sampels.append(item)

        return data_sampels
