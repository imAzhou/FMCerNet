import torch
from torch import nn
import torch.nn.functional as F

class MulticlassDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(MulticlassDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        logits: (batch_size, num_classes) - 模型的原始输出
        targets: (batch_size) - 真实标签索引
        """
        num_classes = logits.shape[1]
        
        # 1. 对 logits 进行 softmax 得到概率
        probs = F.softmax(logits, dim=1)
        
        # 2. 将 targets 转换为 one-hot 编码 (batch_size, num_classes)
        targets_one_hot = F.one_hot(targets, num_classes).float()
        
        # 3. 计算 Intersection (分子) 和 Union (分母)
        # dim=0 表示在 batch 维度上聚合
        intersection = torch.sum(probs * targets_one_hot, dim=0)
        cardinality = torch.sum(probs + targets_one_hot, dim=0)
        
        # 4. 计算每个类别的 Dice Score
        # 2 * intersection / (cardinality + smooth)
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        
        # 5. 取所有类别 Dice 的平均值，然后用 1 减去它得到 Loss
        # 你也可以根据需要改为加权平均
        return 1. - torch.mean(dice_score)


class SupConLoss(nn.Module):
    """
    简化的 Supervised Contrastive Loss，专门适配 (Batch, Dim) 输入。
    """
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels=None, mask=None):
        """
        Args:
            features: hidden vector of shape [bsz, n_dim].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j is the same class as sample i.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0]

        # 1. L2 Normalize (非常关键，对比学习必须在单位球面上做)
        features = F.normalize(features, dim=1)

        # 2. 计算 Logits (Cosine Similarity Matrix)
        # shape: [bsz, bsz]
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)

        # 3. 数值稳定性处理 (减去每行的最大值，防止 exp 溢出)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # 4. 构建 Mask
        if labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            # mask[i][j] = 1 if labels[i] == labels[j]
            mask = torch.eq(labels, labels.T).float().to(device)
        elif mask is None:
            # 如果既没有 labels 也没有 mask，抛出错误
            raise ValueError("SupConLoss needs either 'labels' or 'mask'.")
        else:
            mask = mask.to(device)

        # 5. Mask-out self-contrast (将对角线设为 0)
        # 也就是自己跟自己不算“正样本”，也不算“负样本”，直接踢出计算
        # 这里用 fill_diagonal_ 替代了原本报错的 scatter，逻辑更清晰
        logits_mask = torch.ones_like(mask).to(device)
        logits_mask.fill_diagonal_(0) # 对角线置0
        
        mask = mask * logits_mask # 确保 mask 对角线也是 0

        # 6. 计算 Log Probability
        # 分母：sum over all k != i (所有非自身的样本都作为分母项)
        exp_logits = torch.exp(logits) * logits_mask
        # sum(1) 对行求和
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        # 7. 计算 Mean Log-Likelihood over Positive Pairs
        # 分子：只取 mask=1 的项 (正样本对)
        # mask.sum(1) 是每个样本对应的正样本数量
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)

        # 8. Loss 计算
        # 如果某个样本没有正样本对 (mask.sum(1)==0)，它的 loss 会是 0 (因为 mask*log_prob 也是0)
        # 我们只对那些有正样本对的样本求平均
        # 但是为了代码简单，通常直接取 mean，因为 mask=0 的项分子分母都是 0 (加上了 epsilon)
        loss = - mean_log_prob_pos
        loss = loss.mean()

        return loss
    