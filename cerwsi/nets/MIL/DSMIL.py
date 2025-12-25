from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mil_meta import MIL
from .layers import create_mlp


# --- Core Model Components ---

class IClassifier(nn.Module):
    """Instance-level classifier."""

    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.inst_classifier = nn.Linear(in_dim, num_classes)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B x M x D]
        c = self.inst_classifier(h)  # B x M x C
        return c


class BClassifier(nn.Module):
    """Bag-level classifier with attention."""

    def __init__(self, in_dim: int, attn_dim: int = 384, dropout: float = 0.0):
        super().__init__()
        self.q = nn.Linear(in_dim, attn_dim)
        self.v = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, in_dim)
        )
        self.norm = nn.LayerNorm(in_dim)
        self.fcc = nn.Conv1d(3, 3, kernel_size=in_dim)

    def forward(self, h: torch.Tensor, c: torch.Tensor, attn_mask=None) -> tuple[torch.Tensor, torch.Tensor]:
        device = h.device
        V = self.v(h)  # B x M x D
        Q = self.q(h)  # B x M x D_attn

        # Sort instances by class scores to find critical instances
        _, m_indices = torch.sort(c, dim=1, descending=True)

        # Select features of top instances for each class
        m_feats = torch.stack(
            [torch.index_select(h_i, dim=0, index=m_indices_i[0, :]) for h_i, m_indices_i in zip(h, m_indices)], 0
        )

        q_max = self.q(m_feats)  # B x C x D_attn
        # Attention mechanism: I think this could be the error?
        A = torch.bmm(Q, q_max.transpose(1, 2))  # B x M x C
        if attn_mask is not None:
            A = A + (1 - attn_mask).unsqueeze(dim=2) * torch.finfo(A.dtype).min

        A = F.softmax(A / torch.sqrt(torch.tensor(Q.shape[-1], dtype=torch.float32, device=device)),
                      dim=1)  # Softmax over M

        # Aggregate features

        B = torch.bmm(A.transpose(1, 2), V)  # B x C x D

        B = self.norm(B)
        return B, A


# --- Main DSMIL Module (inherits from MIL base) ---

class DSMIL(MIL):
    def __init__(
            self,
            in_dim: int = 1024,
            embed_dim: int = 512,
            num_fc_layers: int = 1,
            dropout: float = 0.25,
            attn_dim: int = 384,
            dropout_v: float = 0.0,
            num_classes: int = 2,
            classes = None,
    ):
        super().__init__(in_dim=in_dim, embed_dim=embed_dim, num_classes=num_classes, classes=classes)
        self.patch_embed = create_mlp(
            in_dim=in_dim,
            hid_dims=[embed_dim] * (num_fc_layers - 1),
            out_dim=embed_dim,
            dropout=dropout,
            end_with_fc=False
        )
        self.i_classifier = IClassifier(in_dim=embed_dim, num_classes=num_classes)
        self.b_classifier = BClassifier(in_dim=embed_dim, attn_dim=attn_dim, dropout=dropout_v)
        self.classifier = nn.Conv1d(num_classes, num_classes, kernel_size=embed_dim)
        self.initialize_weights()

    def forward_features(self, h: torch.Tensor, attn_mask=None, return_attention: bool = False) -> tuple[
        torch.Tensor, dict]:
        h = self.patch_embed(h)
        instance_classes = self.i_classifier(h)
        slide_feats, attention = self.b_classifier(h, instance_classes, attn_mask=attn_mask)
        intermeds = {'instance_classes': instance_classes}
        if return_attention:
            intermeds['attention'] = attention

        return slide_feats, intermeds

    def forward_attention(self, h: torch.Tensor, attn_mask=None, attn_only=True) -> torch.Tensor:
        pass

    def initialize_classifier(self, num_classes: Optional[int] = None):
        self.classifier = nn.Conv1d(num_classes, num_classes, kernel_size=self.embed_dim)

    def forward_head(self, slide_feats: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(slide_feats)  # B x C x 1
        return logits.squeeze(-1)

    def forward(self, h: torch.Tensor, label: torch.LongTensor = None, loss_fn: nn.Module = None,
                attn_mask=None, return_attention: bool = False,
                return_slide_feats: bool = False) -> tuple[dict, dict]:
        slide_feats, intermeds = self.forward_features(h, attn_mask=attn_mask, return_attention=return_attention)
        max_instance_logits, _ = torch.max(intermeds['instance_classes'], 1)
        bag_logits = self.forward_head(slide_feats)
        logits = 0.5 * (bag_logits + max_instance_logits)
        cls_loss = self.compute_loss(loss_fn, logits, label)

        results_dict = {'logits': logits, 'loss': cls_loss}
        log_dict = {'loss': cls_loss.item() if cls_loss is not None else -1}
        if not return_attention and 'attention' in log_dict:
            del log_dict['attention']
        if return_slide_feats:
            log_dict['slide_feats'] = slide_feats

        return results_dict, log_dict

    def calc_loss(self, databatch):
        input_x = databatch['inputs'].to(self.device)   # (bs, k, c)
        attn_mask = torch.stack([item.attn_mask for item in databatch['data_samples']]).to(self.device)
        slide_feats, intermeds = self.forward_features(input_x, attn_mask=attn_mask, return_attention=False)
        max_instance_logits, _ = torch.max(intermeds['instance_classes'], 1)
        bag_logits = self.forward_head(slide_feats)
        pred_logits = 0.5 * (bag_logits + max_instance_logits)

        loss_fn = nn.CrossEntropyLoss()
        label = torch.as_tensor([item.slide_label for item in databatch['data_samples']]).to(self.device)
        loss = loss_fn(pred_logits, label)
        return loss, {'bce_loss': loss}
    
    def set_pred(self, databatch):
        input_x = databatch['inputs'].to(self.device)   # (bs, k, c)
        attn_mask = torch.stack([item.attn_mask for item in databatch['data_samples']]).to(self.device)
        slide_feats, intermeds = self.forward_features(input_x, attn_mask=attn_mask, return_attention=False)
        max_instance_logits, _ = torch.max(intermeds['instance_classes'], 1)
        bag_logits = self.forward_head(slide_feats)
        pred_logits = 0.5 * (bag_logits + max_instance_logits)

        pred_labels = torch.argmax(pred_logits, dim=1)
        pred_probs = nn.functional.softmax(pred_logits, dim = 1)
        data_sampels = []
        for item, label, prob in zip(databatch['data_samples'], pred_labels, pred_probs):
            item.pred_label = label
            item.pred_clsname = self.classes[label]
            item.pred_prob = prob
            data_sampels.append(item)

        return data_sampels
