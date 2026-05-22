import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import GlobalAttention, GlobalGatedAttention, create_mlp
from .mil_meta import MIL



class ABMIL(MIL):
    """
    ABMIL (Attention-based Multiple Instance Learning) model.

    This class implements the core ABMIL architecture, which uses a patch embedding MLP,
    followed by a global attention or gated attention mechanism, and an optional classification head.

    Args:
        in_dim (int): Input feature dimension for each instance (default: 1024).
        embed_dim (int): Embedding dimension after patch embedding (default: 512).
        num_fc_layers (int): Number of fully connected layers in the patch embedding MLP (default: 1).
        dropout (float): Dropout rate applied in the MLP and attention layers (default: 0.25).
        attn_dim (int): Dimension of the attention mechanism (default: 384).
        gate (int): Whether to use gated attention (True) or standard attention (False) (default: True).
        num_classes (int): Number of output classes for the classification head (default: 2).
    """

    def __init__(
            self,
            in_dim: int = 1024,
            embed_dim: int = 512,
            num_fc_layers: int = 1,
            dropout: float = 0.25,
            attn_dim: int = 384,
            gate: int = True,
            num_classes: int = 2,
            classes = None,
    ):
        super().__init__(in_dim=in_dim, embed_dim=embed_dim, num_classes=num_classes, classes=classes)

        self.patch_embed = create_mlp(
            in_dim=in_dim,
            hid_dims=[embed_dim] *
                     (num_fc_layers - 1),
            dropout=dropout,
            out_dim=embed_dim,
            end_with_fc=False
        )

        attn_func = GlobalGatedAttention if gate else GlobalAttention
        self.global_attn = attn_func(
            L=embed_dim,
            D=attn_dim,
            dropout=dropout,
            num_classes=1
        )

        if num_classes > 0:
            self.classifier = nn.Linear(embed_dim, num_classes)
        self.initialize_weights()

    def forward_attention(self, h: torch.Tensor, attn_mask=None, attn_only=True) -> torch.Tensor:
        """
        Compute the attention scores (and optionally the embedded features) for the input instances.

        Args:
            h (torch.Tensor): Input tensor of shape [B, M, D], where B is the batch size,
                M is the number of instances (patches), and D is the input feature dimension.
            attn_mask (torch.Tensor, optional): Optional attention mask of shape [B, M], where 1 indicates
                valid positions and 0 indicates masked positions. If provided, masked positions are set to
                a very large negative value before softmax.
            attn_only (bool, optional): If True, return only the attention scores (A).
                If False, return a tuple (h, A) where h is the embedded features and A is the attention scores.

        Returns:
            torch.Tensor: If attn_only is True, returns the attention scores tensor of shape [B, K, M],
                where K is the number of attention heads (usually 1). If attn_only is False, returns a tuple
                (h, A) where h is the embedded features of shape [B, M, D'] and A is the attention scores.
        """
        h = self.patch_embed(h)
        A = self.global_attn(h)  # B x M x K
        A = torch.transpose(A, -2, -1)  # B x K x M
        if attn_mask is not None:
            A = A + (1 - attn_mask).unsqueeze(dim=1) * torch.finfo(A.dtype).min

        if attn_only:
            return A
        return h, A

    def forward_features(self, h: torch.Tensor, attn_mask=None, return_attention: bool = True) -> torch.Tensor:
        """
        Compute bag-level features using attention pooling.

        Args:
            h (torch.Tensor): [B, M, D] input features.
            attn_mask (torch.Tensor, optional): Attention mask.

        Returns:
            Tuple[torch.Tensor, dict]: Bag features [B, D] and attention weights.
        """
        h, A_base = self.forward_attention(h, attn_mask=attn_mask, attn_only=False)  # A == B x K x M
        A = F.softmax(A_base, dim=-1)  # softmax over N
        h = torch.bmm(A, h).squeeze(dim=1)  # B x K x C --> B x C
        log_dict = {'attention': A_base if return_attention else None}
        return h, log_dict

    def forward_head(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: [B x D]-dim torch.Tensor.

        Returns:
            logits: [B x num_classes]-dim torch.Tensor.
        """
        logits = self.classifier(h)
        return logits

    def forward(self, h: torch.Tensor,
                loss_fn: nn.Module = None,
                label: torch.LongTensor = None,
                attn_mask=None,
                return_attention: bool = False,
                return_slide_feats: bool = False) -> torch.Tensor:
        """
        Forward pass for ABMIL.

        Args:
            h: [B, M, D] input features.
            loss_fn: Optional loss function.
            label: Optional labels.
            attn_mask: Optional attention mask.

        Returns:
            Tuple of (results_dict, log_dict) with logits and loss.
        """
        wsi_feats, log_dict = self.forward_features(h, attn_mask=attn_mask, return_attention=return_attention)
        logits = self.forward_head(wsi_feats)
        cls_loss = MIL.compute_loss(loss_fn, logits, label)
        results_dict = {'logits': logits, 'loss': cls_loss}
        log_dict['loss'] = cls_loss.item() if cls_loss is not None else -1
        if return_slide_feats:
            log_dict['slide_feats'] = wsi_feats
        return results_dict, log_dict

    def calc_loss(self, databatch):
        input_x = databatch['inputs'].to(self.device)   # (bs, k, c)
        attn_mask = torch.stack([item.attn_mask for item in databatch['data_samples']]).to(self.device)
        wsi_feats, log_dict = self.forward_features(input_x, attn_mask=attn_mask, return_attention=False)
        pred_logits = self.forward_head(wsi_feats)

        loss_fn = nn.CrossEntropyLoss()
        label = torch.as_tensor([item.slide_label for item in databatch['data_samples']]).to(self.device)
        loss = loss_fn(pred_logits, label)
        return loss, {'bce_loss': loss}
    
    def set_pred(self, databatch):
        input_x = databatch['inputs'].to(self.device)   # (bs, k, c)
        attn_mask = torch.stack([item.attn_mask for item in databatch['data_samples']]).to(self.device)
        wsi_feats, log_dict = self.forward_features(input_x, attn_mask=attn_mask, return_attention=False)
        pred_logits = self.forward_head(wsi_feats)
        pred_labels = torch.argmax(pred_logits, dim=1)
        pred_probs = nn.functional.softmax(pred_logits, dim = 1)
        data_sampels = []
        for item, label, prob in zip(databatch['data_samples'], pred_labels, pred_probs):
            item.pred_label = label
            item.pred_clsname = self.classes[label]
            item.pred_prob = prob
            data_sampels.append(item)

        return data_sampels