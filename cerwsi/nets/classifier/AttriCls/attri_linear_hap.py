import torch
from torch import nn
import torch.nn.functional as F
from ..meta_classifier import MetaClassifier
from cerwsi.utils import build_evaluator, AttriMcMetric

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

class AttriHAP(MetaClassifier):
    def __init__(self, args):
        self.attribute_classes = args.attribute_classes
        self.classes = args.classes
        self.num_attributes = len(self.attribute_classes)
        
        # 3. 评测指标保持 AttriMcMetric 不变
        evaluator = build_evaluator([AttriMcMetric(
            num_classes = args.num_classes,
            classes = self.classes,
            logger_name = args.logger_name,
            attribute_names = args.attribute_names,
            attribute_classes = self.attribute_classes,
            num_attributes = self.num_attributes,
        )])
        
        super(AttriHAP, self).__init__(evaluator, args)
        
        input_embed_dim = args.backbone_cfg['backbone_output_dim'][-1]
        
        # HAP 参数: lambda (平衡预测误差) 和 gamma (虽然这里只用属性超图，暂不需要gamma)
        # 这里的 lambda 对应论文公式 (8) 中的 lambda，用于平衡 Regularization 和 Error
        self.loss_lambda = getattr(args, 'hap_lambda', 10.0) 
        
        # 输出维度为所有属性类别数的总和
        self.total_logits_dim = sum(self.attribute_classes)
        
        # 学习投影矩阵 B (即这里的 Mlp head)
        # HAP 论文中是线性投影，这里保留 MLP 结构以增强非线性能力，符合 Deep Learning 实践
        self.head = Mlp(
            in_features = input_embed_dim, 
            hidden_features=input_embed_dim//2, 
            out_features=self.total_logits_dim, 
            act_layer=nn.GELU, 
            drop=0.02
        )

    def calc_logits(self, inputs):
        cls_token = self.get_cls_token(inputs)  # (bs, C)
        # 这里的 output 对应论文中的 F = X^T B (或者 kernel 形式)
        logits_flat = self.head(cls_token)      # (bs, total_logits_dim)
        
        v_prime_list = torch.split(logits_flat, self.attribute_classes, dim=1)
        return v_prime_list, cls_token

    def construct_hypergraph_laplacian(self, H_matrix):
        """
        构建超图拉普拉斯矩阵 L_H
        参考论文公式 (1)-(6)
        H_matrix: (Batch_Size, Num_Hyperedges) - 这里的 Num_Hyperedges 等于总的属性值数量
        """
        # 1. 计算超边度数矩阵 D_e (对角矩阵)
        # delta(e) = sum_{v in e} h(v, e)
        d_e = torch.sum(H_matrix, dim=0) # (Num_Hyperedges,)
        # 避免除以0，添加 epsilon
        d_e_inv = 1.0 / (d_e + 1e-5)
        D_e_inv_mat = torch.diag(d_e_inv)
        
        # 2. 计算顶点度数矩阵 D_v (对角矩阵)
        # 论文公式 (3): d(v) = sum_{e in E} w(e)h(v, e). 
        # 这里简化 w(e)=1 (或者可以视为已被包含在 H 构建中)
        # Zhou's Normalized Laplacian 定义中通常还需要考虑 w(e)
        # 这里使用矩阵形式近似: D_v = H * W * H^T 的行和 (假设 W=I)
        # 简化版: 顶点度数 = 该样本拥有的属性数量
        d_v = torch.sum(H_matrix, dim=1) # (Batch_Size,)
        d_v_inv_sqrt = 1.0 / torch.sqrt(d_v + 1e-5)
        D_v_inv_sqrt_mat = torch.diag(d_v_inv_sqrt)
        
        # 3. 计算归一化拉普拉斯矩阵 L_H
        # 论文公式 (6): L_H = I - D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2}
        # 假设 W 为单位阵 (论文公式4使用了热核权重，在 batch 训练中计算两两距离较重，这里先简化为单位权)
        n_samples = H_matrix.size(0)
        I = torch.eye(n_samples, device=H_matrix.device)
        
        # Term: H * D_e^{-1} * H^T
        term = torch.matmul(torch.matmul(H_matrix, D_e_inv_mat), H_matrix.t())
        
        # Normalized: D_v^{-1/2} * Term * D_v^{-1/2}
        normalized_term = torch.matmul(torch.matmul(D_v_inv_sqrt_mat, term), D_v_inv_sqrt_mat)
        
        L_H = I - normalized_term
        return L_H

    def calc_loss(self, inputs, databatch):
        """
        实现 HAP Loss: Hypergraph Regularization + Prediction Error
        公式 (8): argmin { Tr(F^T L_H F) + lambda ||F - Y||^2 }
        """
        # 1. 获取预测值 F (Prediction)
        pred_logits_list, _ = self.calc_logits(inputs)
        # 拼接所有属性的 logits，形状 (bs, sum(attr_classes))
        F_pred = torch.cat(pred_logits_list, dim=1) 
        
        # 2. 获取真值并构建目标矩阵 Y (Target)
        # attr_gt: (bs, num_attributes)
        attr_gt = torch.stack([item.attr_v for item in databatch['data_samples']]).to(self.device)
        
        # 将 Multi-class 属性标签转换为 One-Hot 形式，作为 H 矩阵
        # 例如属性1有5类，属性2有3类 -> 总维度 8。
        H_list = [
            F.one_hot(attr_gt[:, i], num_classes=num_cls) 
            for i, num_cls in enumerate(self.attribute_classes)
        ]
        H_matrix = torch.cat(H_list, dim=1).float() # (Batch_Size, Total_Attr_Dims)
        
        # 论文中 Y = 2H - 1 (将 {0,1} 映射为 {-1, 1})，以便以 0 为边界
        Y_target = 2 * H_matrix - 1
        
        # 3. 计算 Prediction Error Loss (公式 7)
        # Delta(F, Y) = ||F - Y||^2 (MSE Loss)
        loss_prediction = F.mse_loss(F_pred, Y_target, reduction='mean')
        # 注意：论文是 sum squared error，mse 是 mean，数值量级会有差异，依靠 lambda 调整
        
        # 4. 计算 Hypergraph Regularization Loss (公式 6)
        # Omega(F) = Tr(F^T L_H F)
        # 在 Mini-batch 中构建基于当前 Batch 的超图
        L_H = self.construct_hypergraph_laplacian(H_matrix)
        
        # 计算 Trace(F^T * L_H * F)
        # 优化计算: Tr(A B) = sum(A * B^T). 
        # F^T (D, N) * L (N, N) * F (N, D) -> (D, D) trace
        # 更高效的方法: sum((L @ F) * F)
        L_F = torch.matmul(L_H, F_pred)
        trace_loss = torch.sum(L_F * F_pred) / attr_gt.size(0) # Normalize by batch size for stability
        
        # 总损失
        total_loss = trace_loss + self.loss_lambda * loss_prediction
        
        loss_dict = dict(
            loss=total_loss.item(),
            loss_pred=loss_prediction.item(),
            loss_reg=trace_loss.item()
        )
        return total_loss, loss_dict

    def DAP(self, X_prob, cls_attr_dist):
        """
        Distance-based Attribute Prediction (DAP)
        保持不变，用于从属性预测推断样本类别
        """
        attr_dims = torch.tensor(self.attribute_classes, device=self.device)
        zero_tensor = torch.tensor([0], device=self.device)
        offsets = torch.cat((zero_tensor, torch.cumsum(attr_dims, dim=0)[:-1])) 
        total_dim = attr_dims.sum().item() 

        proto_vec_list = []    
        proto_label_list = []  
        for class_idx, class_name in enumerate(self.classes):
            combinations = cls_attr_dist[class_name]
            comb_tensor = torch.tensor(combinations, device=self.device, dtype=torch.long)
            global_indices = comb_tensor + offsets 
            
            num_k = comb_tensor.size(0)
            k_protos = torch.zeros((num_k, total_dim), device=self.device)
            k_protos.scatter_(dim=1, index=global_indices, value=1.0)
            proto_vec_list.append(k_protos)
            
            k_labels = torch.full((num_k,), class_idx, device=self.device, dtype=torch.long)
            proto_label_list.append(k_labels)
            
        prototypes_tensor = torch.cat(proto_vec_list, dim=0)
        proto_labels_tensor = torch.cat(proto_label_list, dim=0)
        
        scores = torch.matmul(X_prob, prototypes_tensor.t())
        best_match_indices = torch.argmax(scores, dim=1) 
        y_pred = proto_labels_tensor[best_match_indices]
        return y_pred

    def set_pred(self, inputs, databatch):
        """
        预测阶段
        """
        pred_logits_list, _ = self.calc_logits(inputs)
        
        # 1. 属性预测
        # 由于我们使用 MSE 回归到 {-1, 1}，Logits 本身的值越大表示越接近 1 (存在)
        # 所以依然使用 argmax 获取每个属性组中响应最大的类别
        bs_pred_attr_labels = torch.stack([torch.argmax(logits, dim=1) for logits in pred_logits_list], dim=1)
        
        # 2. 准备 DAP 输入
        # HAP 论文中可以直接使用投影值 F (即 logits) 作为置信度
        # 为了配合 DAP (通常基于概率)，我们可以直接传入 logits (因为 DAP 是点积操作，单调性一致即可)
        # 或者进行 sigmoid/softmax 归一化。为了兼容性，这里拼接 logits。
        bs_pred_prob_flatten = torch.cat(pred_logits_list, dim=1)
        
        # 如果需要严格限制在 [0,1] 区间给 DAP，可以使用 Sigmoid:
        # bs_pred_prob_flatten = torch.sigmoid(bs_pred_prob_flatten)
        # 但考虑到 DAP 做点积，原始的有符号 Logits (回归值) 包含了正负信息，效果可能更好。
        
        cls_attr_dist = databatch['data_samples'][0].cls_attr_dist
        
        # 3. 类别预测 (DAP)
        bs_pred_cls_label = self.DAP(bs_pred_prob_flatten, cls_attr_dist)
        
        data_samples = []
        for i, item in enumerate(databatch['data_samples']):
            item.pred_prob_flatten = bs_pred_prob_flatten[i]
            item.pred_attr_label = bs_pred_attr_labels[i]
            item.pred_cls_label = bs_pred_cls_label[i]
            data_samples.append(item)

        return data_samples
    