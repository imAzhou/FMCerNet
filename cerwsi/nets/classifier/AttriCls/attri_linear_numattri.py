import torch
from torch import nn
from ..meta_classifier import MetaClassifier
from cerwsi.utils import build_evaluator, AttriMetric, AttriMcMetric
import torch.nn.functional as F
from .attr_loss import MulticlassDiceLoss,SupConLoss

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
        self.classes = args.classes
        self.num_attributes = len(self.attribute_classes)
        
        evaluator = build_evaluator([AttriMcMetric(
            num_classes = args.num_classes,
            classes = self.classes,
            logger_name = args.logger_name,
            attribute_names = args.attribute_names,
            attribute_classes = self.attribute_classes,
            num_attributes = self.num_attributes,
        )])
        # evaluator = build_evaluator([AttriMetric(
        #     args.logger_name,
        #     attribute_names = args.attribute_names,
        #     attribute_classes = self.attribute_classes,
        #     num_attributes = self.num_attributes,
        # )])
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
        # self.supcon_proj = nn.Sequential(
        #     nn.Linear(input_embed_dim, input_embed_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(input_embed_dim, 256) # 映射到低维空间 (如 128) 效果更好
        # )

    def calc_logits(self, inputs):
        cls_token = self.get_cls_token(inputs)  # (bs, C)
        logits_flat = self.head(cls_token)      # (bs, total_logits_dim)
        
        # 2. 使用 torch.split 将扁平的输出按照每个属性的类别数切分开
        # 返回的是一个 list，每个元素的 shape 为 (bs, num_class_i)
        v_prime_list = torch.split(logits_flat, self.attribute_classes, dim=1)
        return v_prime_list,cls_token

    # def calc_loss(self, inputs, databatch):
    #     # 1. 获取预测值：长度为 num_attributes 的 list，每个元素 shape 为 (bs, cls_i)
    #     pred_logits_list,feat_v = self.calc_logits(inputs)
    #     # features_proj = self.supcon_proj(feat_v)

    #     # 2. 获取真值 (bs, num_attributes)
    #     attr_gt = torch.stack([item.attr_v for item in databatch['data_samples']]).to(self.device)
    #     total_loss = 0
    #     loss_ce_total,loss_dice_total,loss_supcon_total = 0.0, 0.0, 0.0
        
    #     dice_loss_fn = MulticlassDiceLoss()
    #     supcon_loss_fn = SupConLoss(temperature=0.1)
    #     # 3. 循环计算每个属性的 Loss
    #     for i in range(self.num_attributes):
    #         # 获取该属性对应的权重 buffer，并针对当前属性计算加权交叉熵
    #         weight = getattr(self, f'attr_{i}_weights')
    #         ce_loss = F.cross_entropy(
    #             pred_logits_list[i], 
    #             attr_gt[:, i], 
    #             weight=weight
    #         )
    #         lambda_dice,lambda_supcon = 1.0, 1.0
    #         dice_loss = lambda_dice * dice_loss_fn(pred_logits_list[i], attr_gt[:, i])
    #         # supcon_loss = lambda_supcon * supcon_loss_fn(features_proj, labels=attr_gt[:, i])
            
    #         current_attr_loss = ce_loss + dice_loss

    #         total_loss += current_attr_loss
    #         loss_ce_total += ce_loss.item()
    #         loss_dice_total += dice_loss.item()
    #         loss_supcon_total += supcon_loss.item()

    #     loss_dict = dict(
    #         loss=total_loss.item(), 
    #         loss_ce=loss_ce_total, 
    #         loss_dice=loss_dice_total,
    #         loss_supcon=loss_supcon_total
    #     )
    #     return total_loss, loss_dict

    def calc_loss(self, inputs, databatch):
        pred_logits_list,_ = self.calc_logits(inputs)
        all_logits = torch.cat(pred_logits_list, dim=1)
        # 获取真值 (bs, num_attributes)
        attr_gt = torch.stack([item.attr_v for item in databatch['data_samples']]).to(self.device)
        all_targets_list = [
            F.one_hot(attr_gt[:, i], num_classes=num_cls) 
            for i, num_cls in enumerate(self.attribute_classes)
        ]
        all_targets = torch.cat(all_targets_list, dim=1).float()
        total_loss = F.l1_loss(all_logits, all_targets, reduction='mean')
        loss_dict = dict(
            loss=total_loss.item(), 
        )
        return total_loss, loss_dict
    
    # def set_pred(self, inputs, databatch):
    #     pred_logits_list,_ = self.calc_logits(inputs)
    #     # 4. 对每个属性分别取 argmax,结果拼接成 (bs, num_attributes)
    #     pred_labels = torch.stack([torch.argmax(logits, dim=1) for logits in pred_logits_list], dim=1)
    #     data_samples = []
    #     for i, item in enumerate(databatch['data_samples']):
    #         # 存储该样本所有属性的 logits
    #         pred_logits = [logits[i] for logits in pred_logits_list]    # 每个样本在 K 个属性上 M 个类别的 logit值：[[.....],[..],[...]]
    #         pred_prob_flatten = []
    #         for logitlist in pred_logits:
    #             pred_prob_flatten.extend(F.softmax(logitlist, dim=0).tolist())
    #         item.pred_prob_flatten = torch.tensor(pred_prob_flatten)    # tensor: (sum(self.attribute_classes),)
    #         item.pred_label = pred_labels[i]
    #         data_samples.append(item)

    #     return data_samples

    def DAP(self, X_prob, cls_attr_dist):
        """
        Distance-based Attribute Prediction (DAP) - 修正版
        
        Args:
            X_prob: (N, 26) 模型的预测概率向量
            cls_attr_dist: 字典，格式 {'NILM': [[...attr_indices], ...], ...}
            
        Returns:
            y_pred: (N,) 预测的类别索引
        """
        # 1. 准备 One-Hot 映射所需的 Offset
        # offsets = [0, 5, 8, 11, 13, 18, 22, 24]
        attr_dims = torch.tensor(self.attribute_classes, device=self.device)
        zero_tensor = torch.tensor([0], device=self.device)
        offsets = torch.cat((zero_tensor, torch.cumsum(attr_dims, dim=0)[:-1])) # Shape: (8,)
        total_dim = attr_dims.sum().item() # 26
        # 2. 构建属性原型库 (Vectorized Construction)
        proto_vec_list = []    # 存储每个类别的原型 Tensor 块
        proto_label_list = []  # 存储对应的 label Tensor 块
        for class_idx, class_name in enumerate(self.classes):
            # combinations 是一个 list of lists，例如 [[0,2,...], [1,0,...]]
            combinations = cls_attr_dist[class_name]
            comb_tensor = torch.tensor(combinations, device=self.device, dtype=torch.long)
            global_indices = comb_tensor + offsets  # Shape: (K, 8)
            # 创建 One-Hot 容器: (K, 26)
            num_k = comb_tensor.size(0)
            k_protos = torch.zeros((num_k, total_dim), device=self.device)
            # 使用 scatter_ 填充 1: dim=1 表示在列方向 scatter, value=1.0 表示填充的值
            k_protos.scatter_(dim=1, index=global_indices, value=1.0)
            proto_vec_list.append(k_protos)
            # 记录对应的 Label: 创建一个长度为 K 的 tensor，值全为 class_idx
            k_labels = torch.full((num_k,), class_idx, device=self.device, dtype=torch.long)
            proto_label_list.append(k_labels)
        # 将列表拼接成完整的原型矩阵: prototypes_tensor: (Total_K, 26)
        prototypes_tensor = torch.cat(proto_vec_list, dim=0)
        # proto_labels_tensor: (Total_K,)
        proto_labels_tensor = torch.cat(proto_label_list, dim=0)
        # 3. 计算相似度 (Matrix Multiplication)
        # X_prob: (N, 26)   prototypes_tensor.t(): (26, Total_K)    scores: (N, Total_K)
        scores = torch.matmul(X_prob, prototypes_tensor.t())
        # 4. 找到最大相似度对应的索引
        best_match_indices = torch.argmax(scores, dim=1) # (N,)
        # 5. 映射回类别 Label
        y_pred = proto_labels_tensor[best_match_indices]
        return y_pred


    # def set_pred(self, inputs, databatch):
    #     pred_logits_list,_ = self.calc_logits(inputs)
    #     # 4. 对每个属性分别取 argmax,结果拼接成 (bs, num_attributes)
    #     bs_pred_attr_labels = torch.stack([torch.argmax(logits, dim=1) for logits in pred_logits_list], dim=1)
    #     bs_pred_prob_flatten = torch.cat([F.softmax(logits, dim=1) for logits in pred_logits_list], dim=1)
    #     cls_attr_dist = databatch['data_samples'][0].cls_attr_dist
    #     bs_pred_cls_label = self.DAP(bs_pred_prob_flatten, cls_attr_dist)
    #     data_samples = []
    #     for i, item in enumerate(databatch['data_samples']):
    #         item.pred_prob_flatten = bs_pred_prob_flatten[i]
    #         item.pred_attr_label = bs_pred_attr_labels[i]
    #         item.pred_cls_label = bs_pred_cls_label[i]
    #         data_samples.append(item)

    #     return data_samples

    def set_pred(self, inputs, databatch):
        pred_logits_list,_ = self.calc_logits(inputs)
        # 4. 对每个属性分别取 argmax,结果拼接成 (bs, num_attributes)
        bs_pred_attr_labels = torch.stack([torch.argmax(logits, dim=1) for logits in pred_logits_list], dim=1)
        bs_pred_prob_flatten = torch.cat(pred_logits_list, dim=1)
        cls_attr_dist = databatch['data_samples'][0].cls_attr_dist
        bs_pred_cls_label = self.DAP(bs_pred_prob_flatten, cls_attr_dist)
        data_samples = []
        for i, item in enumerate(databatch['data_samples']):
            item.pred_prob_flatten = bs_pred_prob_flatten[i]
            item.pred_attr_label = bs_pred_attr_labels[i]
            item.pred_cls_label = bs_pred_cls_label[i]
            data_samples.append(item)

        return data_samples
    