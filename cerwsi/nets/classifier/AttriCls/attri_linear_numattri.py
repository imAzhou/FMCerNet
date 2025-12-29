import torch
from torch import nn
from ..meta_classifier import MetaClassifier
from cerwsi.utils import build_evaluator, AttriMetric, AttriMcMetric
import torch.nn.functional as F

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = (drop, drop)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class AttriLinear(MetaClassifier):
    def __init__(self, args):
        # 假设 args 中现在传入的是 attribute_classes 列表
        # attribute_classes = [5, 3, 3, 2, 2, 3, 5, 3, 3, 4]
        self.attribute_classes = args.attribute_classes 
        self.num_attributes = len(self.attribute_classes)
        
        # evaluator = build_evaluator([AttriMcMetric(
        #     num_classes = args.num_classes,
        #     classes = args.classes,
        #     logger_name = args.logger_name,
        #     attribute_names = args.attribute_names,
        #     attribute_classes = self.attribute_classes,
        #     num_attributes = self.num_attributes,
        # )])
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
        
        # 1. 修改输出层：输出维度为所有属性类别数的总和
        self.total_logits_dim = sum(self.attribute_classes)
        # self.head = nn.Linear(input_embed_dim, self.total_logits_dim)
        self.head = Mlp(
            in_features = input_embed_dim, 
            hidden_features=input_embed_dim//2, 
            out_features=self.total_logits_dim, 
            act_layer=nn.GELU, 
            drop=0.02
        )

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

    def calc_loss(self, inputs, databatch):
        # 1. 获取预测值：长度为 num_attributes 的 list，每个元素 shape 为 (bs, cls_i)
        pred_logits_list = self.calc_logits(inputs)
        
        # 2. 获取真值 (bs, num_attributes)
        # 建议直接在 data_samples 中预处理好张量，避免在 loss 函数中循环创建 tensor 以提高性能
        attr_gt = torch.stack([item.attr_v for item in databatch['data_samples']]).to(self.device)
        
        total_loss = 0
        loss_dict = {}
        # 3. 循环计算每个属性的 Loss
        for i in range(self.num_attributes):
            # 获取该属性对应的权重 buffer，并针对当前属性计算加权交叉熵
            weight = getattr(self, f'attr_{i}_weights')
            attr_loss = F.cross_entropy(
                pred_logits_list[i], 
                attr_gt[:, i], 
                weight=weight
            )
            total_loss += attr_loss
            # loss_dict[f'loss_attr_{i}'] = attr_loss.item()
            
        loss_dict['loss'] = total_loss.item()
        return total_loss, loss_dict
    
    def set_pred(self, inputs, databatch):
        pred_logits_list = self.calc_logits(inputs)
        # 4. 对每个属性分别取 argmax,结果拼接成 (bs, num_attributes)
        pred_labels = torch.stack([torch.argmax(logits, dim=1) for logits in pred_logits_list], dim=1)
        
        data_samples = []
        for i, item in enumerate(databatch['data_samples']):
            # 存储该样本所有属性的 logits (可选: 合并或保持 list)
            pred_logits = [logits[i] for logits in pred_logits_list]    # 每个样本在 K 个属性上 M 个类别的 logit值：[[.....],[..],[...]]
            pred_prob_flatten = []
            for logitlist in pred_logits:
                pred_prob_flatten.extend(F.softmax(logitlist, dim=0).tolist())
            item.pred_prob_flatten = torch.tensor(pred_prob_flatten)    # tensor: (sum(self.attribute_classes),)
            item.pred_label = pred_labels[i]
            data_samples.append(item)

        return data_samples

    # def set_pred(self, inputs, databatch):
    #     pred_logits_list = self.calc_logits(inputs)
    #     # 4. 对每个属性分别取 argmax,结果拼接成 (bs, num_attributes)
    #     pred_labels = torch.stack([torch.argmax(logits, dim=1) for logits in pred_logits_list], dim=1)
        
    #     data_samples = []
    #     for i, item in enumerate(databatch['data_samples']):
    #         # 存储该样本所有属性的 logits (可选: 合并或保持 list)
    #         pred_logits = [logits[i] for logits in pred_logits_list]    # 每个样本在 K 个属性上 M 个类别的 logit值：[[.....],[..],[...]]
    #         pred_prob_flatten = []
    #         for logitlist in pred_logits:
    #             pred_prob_flatten.extend(F.softmax(logitlist).tolist())
    #         item.pred_prob_flatten = torch.tensor(pred_prob_flatten)    # tensor: (sum(self.attribute_classes),)
    #         item.pred_attr_label = pred_labels[i]
    #         item.pred_cls_label = pred_labels[i][-1]
    #         data_samples.append(item)

    #     return data_samples
    