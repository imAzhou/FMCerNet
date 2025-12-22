import torch
import torch.nn as nn
import torch.nn.functional as F
from ..meta_classifier import MetaClassifier
from cerwsi.utils import build_evaluator, AttriMetric

class AttriLinear(MetaClassifier):
    def __init__(self, args):
        self.attribute_classes = args.attribute_classes
        self.num_attributes = len(self.attribute_classes)
        self.classes = args.classes 
        self.num_classes = args.num_classes 
        
        evaluator = build_evaluator([AttriMetric(
            args.logger_name,
            num_attributes = self.num_attributes
        )])
        super(AttriLinear, self).__init__(evaluator, args)
        
        input_embed_dim = args.backbone_cfg['backbone_output_dim'][-1]
        self.total_logits_dim = sum(self.attribute_classes)
        self.attri_head = nn.Linear(input_embed_dim, self.total_logits_dim)
        self.cls_head = nn.Linear(self.total_logits_dim, self.num_classes)

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

        # 获取索引形式的真值 (bs, num_attributes)
        attr_gt_idx = torch.tensor([item.attr_v for item in databatch['data_samples']], dtype=torch.long, device=self.device)
        attr_loss_fn = nn.BCEWithLogitsLoss()
        total_attr_loss = 0
        # 遍历每个属性计算 BCE Loss
        for i in range(self.num_attributes):
            # 获取当前属性的预测 logits: (bs, num_class_i)
            logits_i = pred_logits_list[i]
            
            # 修改点 2：将索引真值转换为 One-Hot 编码
            # F.one_hot 返回 (bs, num_class_i)，需要转为 float 以匹配 BCE 损失
            target_i = F.one_hot(attr_gt_idx[:, i], num_classes=self.attribute_classes[i]).float()
            
            # 计算 BCE Loss
            attr_loss = attr_loss_fn(logits_i, target_i)
            total_attr_loss += attr_loss
            
        total_loss = total_attr_loss + cls_loss
        loss_dict = {
            'loss': total_loss.item(),
            'attr_loss': total_attr_loss.item(),
            'cls_loss': cls_loss.item(),
        }
        return total_loss, loss_dict

    def set_pred(self, inputs, databatch):
        pred_logits_list,cls_logits = self.calc_logits(inputs)
        pred_cls_labels = torch.argmax(cls_logits, dim=1) # (bs, num_classes)
        
        pred_probs_list = [torch.sigmoid(logits) for logits in pred_logits_list]
        # 预测属性标签取概率最大的索引
        pred_attr_labels = torch.stack([torch.argmax(probs, dim=1) for probs in pred_probs_list], dim=1)
        
        data_samples = []
        for i, item in enumerate(databatch['data_samples']):
            # 存储该样本所有属性的概率 (已经过 Sigmoid)
            item.pred_prob = [probs[i] for probs in pred_probs_list]
            item.pred_attr_label = pred_attr_labels[i]
            item.pred_cls_label = pred_cls_labels[i]
            data_samples.append(item)

        return data_samples
    