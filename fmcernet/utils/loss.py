import torch
import torch.nn as nn


class AsymmetricLossOptimized(nn.Module):
    """Asymmetric loss for multi-label classification.

    This follows the optimized ASL implementation used by Query2Label, with an
    optional reduction mode so it can also be used where the project needs a
    per-class loss matrix.
    """

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-5,
                 disable_torch_grad_focal_loss=False, reduction='mean',
                 loss_scale=1000.0):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.reduction = reduction
        self.loss_scale = loss_scale

        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f"Unsupported reduction: {reduction}")

    def forward(self, x, y):
        """Calculate ASL loss.

        Args:
            x: Input logits, shape (bs, num_classes).
            y: Multi-label binary targets, shape (bs, num_classes).
        """
        targets = y
        anti_targets = 1 - y

        xs_pos = torch.sigmoid(x)
        xs_neg = 1.0 - xs_pos

        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        loss = targets * torch.log(xs_pos.clamp(min=self.eps))
        loss = loss + anti_targets * torch.log(xs_neg.clamp(min=self.eps))

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                with torch.no_grad():
                    xs_pos_focus = xs_pos * targets
                    xs_neg_focus = xs_neg * anti_targets
                    asymmetric_w = torch.pow(
                        1 - xs_pos_focus - xs_neg_focus,
                        self.gamma_pos * targets + self.gamma_neg * anti_targets)
            else:
                xs_pos_focus = xs_pos * targets
                xs_neg_focus = xs_neg * anti_targets
                asymmetric_w = torch.pow(
                    1 - xs_pos_focus - xs_neg_focus,
                    self.gamma_pos * targets + self.gamma_neg * anti_targets)
            loss = loss * asymmetric_w

        loss = -loss
        if self.reduction == 'none':
            return loss
        if self.reduction == 'sum':
            return loss.sum() / x.size(0) * self.loss_scale
        return loss.sum() / x.size(0) / y.size(1) * self.loss_scale


def build_loss(loss_cfg, **overrides):
    loss_type = loss_cfg['type']
    loss_args = {k: v for k, v in loss_cfg.items() if k != 'type'}
    loss_args.update(overrides)
    if loss_type == 'BCEWithLogitsLoss':
        return nn.BCEWithLogitsLoss(**loss_args)
    if loss_type == 'AsymmetricLossOptimized':
        return AsymmetricLossOptimized(**loss_args)
    raise ValueError(f"Unsupported loss type: {loss_type}")

def contrastive_loss(features, labels, temperature=0.07):
    """
    计算监督对比学习损失。
    
    Args:
        features: Tensor of shape (bs*num_tokens, embed_dim).
        labels: Tensor of shape (bs*num_tokens) with integer class labels.
        temperature: Temperature scaling factor for contrastive loss.
    
    Returns:
        loss: Supervised contrastive loss (scalar).
    """
    # Step 1: Compute pairwise similarity (dot product)
    similarity_matrix = torch.matmul(features, features.T)  # (bs * num_tokens, bs * num_tokens)
    exp_sim = torch.exp(similarity_matrix / temperature)
    
    # Step 2: Create a mask to identify positive pairs
    label_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()  # (bs * num_tokens, bs * num_tokens)
    mask_self = torch.eye(label_mask.size(0), device=features.device)
    positive_mask = label_mask - mask_self  # Remove diagonal (self-pairs)

    # Step 3: Compute log probabilities for positive pairs
    log_prob = similarity_matrix / temperature - torch.log(exp_sim.sum(dim=1, keepdim=True))
    
    # Step 4: Sum log probabilities for positive samples
    positive_log_prob = positive_mask * log_prob
    positive_count = positive_mask.sum(dim=1)
    
    loss = -(positive_log_prob.sum(dim=1) / positive_count.clamp(min=1))  # Avoid division by zero
    return loss.mean()
