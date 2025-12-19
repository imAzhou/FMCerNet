import torch
import torch.nn as nn
import torch.nn.functional as F
from ..meta_classifier import MetaClassifier
from cerwsi.utils import build_evaluator, AttriMetric

class AttriLinear(MetaClassifier):
    def __init__(self, args):
        self.attribute_classes = args.attribute_classes 
        self.num_attributes = len(self.attribute_classes)
        
        evaluator = build_evaluator([AttriMetric(
            args.logger_name,
            num_attributes = self.num_attributes
        )])
        super(AttriLinear, self).__init__(evaluator, args)
        
        input_embed_dim = args.backbone_cfg['backbone_output_dim'][-1]
        
        # 输出层保持不变：总维度依然是所有属性类别之和
        self.total_logits_dim = sum(self.attribute_classes)
        self.head = nn.Linear(input_embed_dim, self.total_logits_dim)
        
        # 修改点 1：使用 BCEWithLogitsLoss (内部自带 Sigmoid)
        # 如果不同属性的重要性不同，也可以在这里传入 weight
        self.loss_fn = nn.BCEWithLogitsLoss()

    def calc_logits(self, inputs):
        cls_token = self.get_cls_token(inputs) 
        logits_flat = self.head(cls_token)      
        v_prime_list = torch.split(logits_flat, self.attribute_classes, dim=1)
        return v_prime_list
    
    def calc_loss(self, inputs, databatch):
        pred_logits_list = self.calc_logits(inputs)
        
        # 获取索引形式的真值 (bs, num_attributes)
        attr_gt_idx = torch.tensor([item.attr_v for item in databatch['data_samples']], dtype=torch.long, device=self.device)
        
        total_loss = 0
        
        # 遍历每个属性计算 BCE Loss
        for i in range(self.num_attributes):
            # 获取当前属性的预测 logits: (bs, num_class_i)
            logits_i = pred_logits_list[i]
            
            # 修改点 2：将索引真值转换为 One-Hot 编码
            # F.one_hot 返回 (bs, num_class_i)，需要转为 float 以匹配 BCE 损失
            target_i = F.one_hot(attr_gt_idx[:, i], num_classes=self.attribute_classes[i]).float()
            
            # 计算 BCE Loss
            attr_loss = self.loss_fn(logits_i, target_i)
            total_loss += attr_loss
            
        loss_dict = {
            'loss': total_loss.item(),
        }
        return total_loss, loss_dict

    def set_pred(self, inputs, databatch):
        pred_logits_list = self.calc_logits(inputs)
        
        # 修改点 3：使用 Sigmoid 计算每个类别的独立概率
        # 虽然使用 argmax 找最大值在 Sigmoid 后结果不变，但存储时通常保存 Sigmoid 后的值
        pred_probs_list = [torch.sigmoid(logits) for logits in pred_logits_list]
        
        # 预测标签依然取概率最大的索引
        pred_labels = torch.stack([torch.argmax(probs, dim=1) for probs in pred_probs_list], dim=1)
        
        data_samples = []
        for i, item in enumerate(databatch['data_samples']):
            # 存储该样本所有属性的概率 (已经过 Sigmoid)
            item.pred_prob = [probs[i] for probs in pred_probs_list]
            item.pred_label = pred_labels[i]
            data_samples.append(item)

        return data_samples
    