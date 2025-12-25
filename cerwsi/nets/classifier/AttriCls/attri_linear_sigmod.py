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
            attribute_names = args.attribute_names,
            attribute_classes = self.attribute_classes,
            num_attributes = self.num_attributes,
        )])
        super(AttriLinear, self).__init__(evaluator, args)
        
        input_embed_dim = args.backbone_cfg['backbone_output_dim'][-1]
        for i, w in enumerate(args.custom_weights):
            self.register_buffer(f'attr_{i}_weights', torch.tensor(w, dtype=torch.float))
        
        self.total_logits_dim = sum(self.attribute_classes)
        self.attri_head = nn.Linear(input_embed_dim, self.total_logits_dim)

    def calc_logits(self, inputs):
        cls_token = self.get_cls_token(inputs) 
        attri_logits_flat = self.attri_head(cls_token)
        v_prime_list = torch.split(attri_logits_flat, self.attribute_classes, dim=1)
        return v_prime_list
    
    def calc_loss(self, inputs, databatch):
        pred_logits_list = self.calc_logits(inputs)
        # 获取索引形式的真值 (bs, num_attributes)
        attr_gt_idx = torch.tensor([item.attr_v for item in databatch['data_samples']], dtype=torch.long, device=self.device)

        total_attr_loss = 0
        # 遍历每个属性计算 BCE Loss
        for i in range(self.num_attributes):
            # 获取当前属性的预测 logits: (bs, num_class_i)
            logits_i = pred_logits_list[i]
            
            # 修改点 2：将索引真值转换为 One-Hot 编码
            # F.one_hot 返回 (bs, num_class_i)，需要转为 float 以匹配 BCE 损失
            target_i = F.one_hot(attr_gt_idx[:, i], num_classes=self.attribute_classes[i]).float()
            weight_i = getattr(self, f'attr_{i}_weights')
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=weight_i)
            # 计算 BCE Loss
            attr_loss = loss_fn(logits_i, target_i)
            total_attr_loss += attr_loss
            
        loss_dict = {
            'loss': total_attr_loss.item(),
        }
        return total_attr_loss, loss_dict

    def set_pred(self, inputs, databatch):
        pred_logits_list = self.calc_logits(inputs)
        
        pred_probs_list = [torch.sigmoid(logits) for logits in pred_logits_list]
        # 预测属性标签取概率最大的索引
        pred_attr_labels = torch.stack([torch.argmax(probs, dim=1) for probs in pred_probs_list], dim=1)
        
        data_samples = []
        for i, item in enumerate(databatch['data_samples']):
            # 存储该样本所有属性的概率 (已经过 Sigmoid)
            # pred_prob = [probs[i] for probs in pred_probs_list]
            # pred_prob_flatten = []
            # for problist in pred_prob:
            #     pred_prob_flatten.extend(problist.tolist())
            # item.pred_prob_flatten = torch.tensor(pred_prob_flatten)

            pred_logits = [logits[i] for logits in pred_logits_list]
            pred_prob_flatten = []
            for problist in pred_logits:
                pred_prob_flatten.extend(problist.tolist())
            item.pred_prob_flatten = torch.tensor(pred_prob_flatten)
            item.pred_label = pred_attr_labels[i]
            data_samples.append(item)

        return data_samples
    