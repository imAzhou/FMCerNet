import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from nystrom_attention import NystromAttention

from .mil_meta import MIL


class TransLayer(nn.Module):
    def __init__(self, dim=512, num_heads=8):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // num_heads,
            heads=num_heads,
            num_landmarks=dim // 2,
            pinv_iterations=6,
            residual=True,
            dropout=0.1,
        )

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.norm(x), mask=attn_mask)
        return x


class CAM(nn.Module):
    def __init__(self, n_channel, mlp_r=2, temperature=1.0):
        super().__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Sequential(
            nn.Linear(n_channel, n_channel // mlp_r),
            nn.ReLU(),
            nn.Linear(n_channel // mlp_r, 3 * n_channel),
        )
        self.temperature = temperature

    def forward(self, x):
        b, c, _, _ = x.shape
        max_feat = self.linear(self.max_pool(x).view(b, c)).view(b, 3 * c, 1, 1)
        avg_feat = self.linear(self.avg_pool(x).view(b, c)).view(b, 3 * c, 1, 1)
        y = torch.sigmoid((max_feat + avg_feat) * self.temperature)
        return y[:, :c], y[:, c:2 * c], y[:, 2 * c:]


class MCAB(nn.Module):
    def __init__(self, dim=512, temperature=3):
        super().__init__()
        self.local_feature3 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=3 // 2, groups=dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, groups=1),
        )
        self.local_feature5 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=5 // 2, groups=dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, groups=1),
        )
        self.cam = CAM(n_channel=dim, mlp_r=2, temperature=temperature)

    def forward(self, x):
        b, n, c = x.shape
        h = int(math.ceil(math.sqrt(n)))
        w = h
        add_length = h * w - n
        if add_length > 0:
            zero_tensor = x.new_zeros(b, add_length, c)
            x_pad = torch.cat([x, zero_tensor], dim=1)
        else:
            x_pad = x

        cnn_feat = x_pad.transpose(1, 2).view(b, c, h, w)
        feat3 = self.local_feature3(cnn_feat)
        feat5 = self.local_feature5(cnn_feat)
        y1, y2, y3 = self.cam(cnn_feat + feat3 + feat5)
        x = y1 * cnn_feat + y2 * feat3 + y3 * feat5
        x = x.flatten(2).transpose(1, 2)
        return x[:, :n]


class CAMILBlock(nn.Module):
    def __init__(self, dim=512, temperature=3, num_heads=8):
        super().__init__()
        self.trans_layer = TransLayer(dim=dim, num_heads=num_heads)
        self.mcab = MCAB(dim=dim, temperature=temperature)

    def forward(self, x, attn_mask=None):
        x = self.trans_layer(x, attn_mask=attn_mask)
        x = self.mcab(x)
        return x


class AttentionNet(nn.Module):
    def __init__(self, in_dim=512, attn_dim=256, dropout=0.0, gate=True):
        super().__init__()
        if gate:
            self.attention_a = nn.Sequential(
                nn.Linear(in_dim, attn_dim),
                nn.Tanh(),
                nn.Dropout(dropout),
            )
            self.attention_b = nn.Sequential(
                nn.Linear(in_dim, attn_dim),
                nn.Sigmoid(),
                nn.Dropout(dropout),
            )
            self.attention_c = nn.Linear(attn_dim, 1)
        else:
            self.attention = nn.Sequential(
                nn.Linear(in_dim, attn_dim),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Linear(attn_dim, 1),
            )
        self.gate = gate

    def forward(self, x):
        if self.gate:
            a = self.attention_a(x)
            b = self.attention_b(x)
            return self.attention_c(a.mul(b))
        return self.attention(x)


class CAMIL(MIL):
    def __init__(
            self,
            in_dim=517,
            embed_dim=512,
            num_classes=2,
            classes=None,
            temperature=1.2,
            dropout=0.15,
            n_layers=4,
            attn_dim=256,
            gate=True,
            num_heads=8,
    ):
        super().__init__(in_dim=in_dim, embed_dim=embed_dim, num_classes=num_classes, classes=classes)
        self.patch_embed = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.layers = nn.ModuleList([
            CAMILBlock(dim=embed_dim, temperature=temperature, num_heads=num_heads)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.attention_net = AttentionNet(
            in_dim=embed_dim,
            attn_dim=attn_dim,
            dropout=0.0,
            gate=gate,
        )
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.initialize_weights()

    def _format_attn_mask(self, attn_mask, h):
        if attn_mask is None:
            return None
        attn_mask = attn_mask.to(device=h.device)
        if attn_mask.shape != h.shape[:2]:
            raise ValueError(f'CAMIL attn_mask expects shape {tuple(h.shape[:2])}, got {tuple(attn_mask.shape)}.')
        attn_mask = attn_mask.bool()
        if torch.any(attn_mask.sum(dim=1) == 0):
            raise ValueError('CAMIL received a sample with no valid tiles in attn_mask.')
        return attn_mask

    def _apply_mask(self, h, attn_mask):
        if attn_mask is None:
            return h
        return h * attn_mask.unsqueeze(-1).to(dtype=h.dtype)

    def forward_attention(self, h, attn_mask=None, attn_only=True):
        h = self.patch_embed(h)
        attn_mask = self._format_attn_mask(attn_mask, h)
        h = self._apply_mask(h, attn_mask)
        for layer in self.layers:
            h = layer(h, attn_mask=attn_mask)
            h = self._apply_mask(h, attn_mask)
        h = self.norm(h)
        h = self._apply_mask(h, attn_mask)
        attention = self.attention_net(h).transpose(1, 2)
        if attn_mask is not None:
            attention = attention.masked_fill(~attn_mask.unsqueeze(1), torch.finfo(attention.dtype).min)
        if attn_only:
            return attention
        return h, attention

    def forward_features(self, h, attn_mask=None, return_attention=False):
        h, attention_logits = self.forward_attention(h, attn_mask=attn_mask, attn_only=False)
        attention = F.softmax(attention_logits, dim=-1)
        wsi_feats = torch.bmm(attention, h).squeeze(1)
        log_dict = {'attention': attention_logits if return_attention else None}
        return wsi_feats, log_dict

    def forward_head(self, h):
        return self.classifier(h)

    def forward(
            self,
            h,
            loss_fn=None,
            label=None,
            attn_mask=None,
            return_attention=False,
            return_slide_feats=False,
    ):
        wsi_feats, log_dict = self.forward_features(h, attn_mask=attn_mask, return_attention=return_attention)
        logits = self.forward_head(wsi_feats)
        cls_loss = MIL.compute_loss(loss_fn, logits, label)
        results_dict = {'logits': logits, 'loss': cls_loss}
        log_dict['loss'] = cls_loss.item() if cls_loss is not None else -1
        if return_slide_feats:
            log_dict['slide_feats'] = wsi_feats
        return results_dict, log_dict

    def calc_loss(self, databatch):
        input_x = databatch['inputs'].to(self.device)
        attn_mask = torch.stack([item.attn_mask for item in databatch['data_samples']]).to(self.device)
        wsi_feats, _ = self.forward_features(input_x, attn_mask=attn_mask, return_attention=False)
        pred_logits = self.forward_head(wsi_feats)

        loss_fn = nn.CrossEntropyLoss()
        label = torch.as_tensor([item.slide_label for item in databatch['data_samples']]).to(self.device)
        loss = loss_fn(pred_logits, label)
        return loss, {'bce_loss': loss}

    def set_pred(self, databatch):
        input_x = databatch['inputs'].to(self.device)
        attn_mask = torch.stack([item.attn_mask for item in databatch['data_samples']]).to(self.device)
        wsi_feats, _ = self.forward_features(input_x, attn_mask=attn_mask, return_attention=False)
        pred_logits = self.forward_head(wsi_feats)

        pred_labels = torch.argmax(pred_logits, dim=1)
        pred_probs = nn.functional.softmax(pred_logits, dim=1)
        data_sampels = []
        for item, label, prob in zip(databatch['data_samples'], pred_labels, pred_probs):
            item.pred_label = label
            item.pred_clsname = self.classes[label]
            item.pred_prob = prob
            data_sampels.append(item)

        return data_sampels
