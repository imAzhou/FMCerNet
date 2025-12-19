import torch
from torch import nn
from ..meta_classifier import MetaClassifier
from cerwsi.utils import build_evaluator, AttriMetric


class AttriLinear(MetaClassifier):
    def __init__(self, args):
        evaluator = build_evaluator([AttriMetric(
            args.logger_name,
            num_attributes = args.num_attributes,
        )])
        super(AttriLinear, self).__init__(evaluator, args)
        input_embed_dim = args.backbone_cfg['backbone_output_dim'][-1]
        self.num_classes = args.num_classes
        self.num_attributes = args.num_attributes
        self.head = nn.Linear(input_embed_dim, self.num_attributes * self.num_classes)

        
    def calc_logits(self, inputs):
        cls_token = self.get_cls_token(inputs)  # (bs, C)
        logits_flat = self.head(cls_token)
        # (bs, num_attributes, num_classes)
        v_prime = logits_flat.view(-1, self.num_attributes, self.num_classes)
        return v_prime
    
    def calc_loss(self, inputs, databatch):
        pred_logits = self.calc_logits(inputs)
        attr_gt = torch.tensor([item.attr_v for item in databatch['data_samples']], 
                               dtype=torch.long, device=self.device)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(pred_logits.view(-1, 5), attr_gt.view(-1))
        loss_dict = {
            'loss': loss.item(),
        }
        return loss,loss_dict

    def set_pred(self, inputs, databatch):
        pred_logits = self.calc_logits(inputs) # (bs, num_attributes, num_classes)
        pred_labels = torch.argmax(pred_logits, dim=2) # (bs, num_attributes)
        data_sampels = []
        for item, logits, labels in zip(databatch['data_samples'], pred_logits, pred_labels):
            item.pred_logit = logits
            item.pred_label = labels
            data_sampels.append(item)

        return data_sampels

