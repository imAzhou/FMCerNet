import torch
from torch import nn
from ..meta_classifier import MetaClassifier
from cerwsi.utils import build_evaluator, AttriMetric


class AttriLinear(MetaClassifier):
    def __init__(self, args):
        # 假设 args 中现在传入的是 attribute_classes 列表
        # attribute_classes = [5, 3, 3, 2, 2, 3, 5, 3, 3, 4]
        self.attribute_classes = args.attribute_classes 
        self.num_attributes = len(self.attribute_classes)
        
        evaluator = build_evaluator([AttriMetric(
            args.logger_name,
            num_attributes = self.num_attributes
        )])
        super(AttriLinear, self).__init__(evaluator, args)
        
        input_embed_dim = args.backbone_cfg['backbone_output_dim'][-1]
        
        # 1. 修改输出层：输出维度为所有属性类别数的总和
        self.total_logits_dim = sum(self.attribute_classes)
        self.head = nn.Linear(input_embed_dim, self.total_logits_dim)

    def calc_logits(self, inputs):
        cls_token = self.get_cls_token(inputs)  # (bs, C)
        logits_flat = self.head(cls_token)      # (bs, total_logits_dim)
        
        # 2. 使用 torch.split 将扁平的输出按照每个属性的类别数切分开
        # 返回的是一个 list，每个元素的 shape 为 (bs, num_class_i)
        v_prime_list = torch.split(logits_flat, self.attribute_classes, dim=1)
        return v_prime_list
    
    def calc_loss(self, inputs, databatch):
        # pred_logits 现在是一个长度为 num_attributes 的 list
        pred_logits_list = self.calc_logits(inputs)
        
        # 获取真值 (bs, num_attributes)
        attr_gt = torch.tensor([item.attr_v for item in databatch['data_samples']], dtype=torch.long, device=self.device)
        
        loss_fn = nn.CrossEntropyLoss()
        total_loss = 0
        
        # 3. 循环计算每个属性的 Loss
        for i in range(self.num_attributes):
            # 取第 i 个属性的预测值 (bs, cls_i) 和 真值 (bs,)
            attr_loss = loss_fn(pred_logits_list[i], attr_gt[:, i])
            total_loss += attr_loss
            
        loss_dict = {
            'loss': total_loss.item(),
        }
        return total_loss, loss_dict

    def set_pred(self, inputs, databatch):
        pred_logits_list = self.calc_logits(inputs)
        
        # 4. 对每个属性分别取 argmax
        # 结果拼接成 (bs, num_attributes)
        pred_labels = torch.stack([torch.argmax(logits, dim=1) for logits in pred_logits_list], dim=1)
        
        data_samples = []
        # 为了方便后续处理，将 list of logits 转换为便于存储的格式或保持原样
        for i, item in enumerate(databatch['data_samples']):
            # 存储该样本所有属性的 logits (可选: 合并或保持 list)
            item.pred_logit = [logits[i] for logits in pred_logits_list]
            item.pred_label = pred_labels[i]
            data_samples.append(item)

        return data_samples
    