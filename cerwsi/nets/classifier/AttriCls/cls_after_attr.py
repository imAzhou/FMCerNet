import torch
import torch.nn as nn
import torch.nn.functional as F
from ..meta_classifier import MetaClassifier
from cerwsi.utils import build_evaluator, MultiClsMetric

class AttriLinear(MetaClassifier):
    def __init__(self, args):
        self.attribute_classes = args.attribute_classes
        self.num_attributes = len(self.attribute_classes)
        self.classes = args.classes 
        self.num_classes = args.num_classes 
        
        evaluator = build_evaluator([MultiClsMetric(
            num_classes = args.num_classes,
            classes = args.classes,
            logger_name = args.logger_name,
        )])
        super(AttriLinear, self).__init__(evaluator, args)
        
        input_embed_dim = args.backbone_cfg['backbone_output_dim'][-1]
        
        self.total_logits_dim = sum(self.attribute_classes)
        self.attri_head = nn.Linear(input_embed_dim, self.total_logits_dim)
        self.cls_head = nn.Linear(self.total_logits_dim, self.num_classes)

        for name, param in self.attri_head.named_parameters():
            param.requires_grad = False


    def calc_logits(self, inputs):
        cls_token = self.get_cls_token(inputs) 
        attri_logits_flat = self.attri_head(cls_token)
        cls_logits = self.cls_head(attri_logits_flat)
        v_prime_list = torch.split(attri_logits_flat, self.attribute_classes, dim=1)
        return v_prime_list,cls_logits
    
    def calc_loss(self, inputs, databatch):
        pred_logits_list,cls_logits = self.calc_logits(inputs)

        clsid_gt_idx = torch.tensor([item.cls_id for item in databatch['data_samples']], dtype=torch.long, device=self.device)
        cls_loss_fn = nn.CrossEntropyLoss()
        cls_loss = cls_loss_fn(cls_logits, clsid_gt_idx)

        loss_dict = {
            'loss': cls_loss.item(),
        }
        return cls_loss, loss_dict

    def set_pred(self, inputs, databatch):
        pred_logits_list,cls_logits = self.calc_logits(inputs)
        pred_probs = torch.softmax(cls_logits, dim=-1) # (bs, n_cls)
        _,pred_labels = torch.max(pred_probs, dim=-1)
        
        data_samples = []
        for i, item in enumerate(databatch['data_samples']):
            # 存储该样本所有属性的概率 (已经过 Sigmoid)
            item.pred_prob = pred_probs[i]
            item.pred_label = pred_labels[i]
            data_samples.append(item)

        return data_samples
    