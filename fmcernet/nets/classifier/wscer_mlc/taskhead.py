import torch
from torch import nn
import math
import torch.nn.functional as F
from .chief import CHIEF
from .twoway_attn import TwoWayAttentionBlock
from ..meta_classifier import MetaClassifier
from fmcernet.utils import build_evaluator, ExtendMultiLabelMetric, build_loss


class MLCQuery(nn.Module):
    def __init__(
        self,
        num_classes,
        input_embed_dim,
        key_gate_scale,
        depth: int=2,
        num_heads: int=8,
        mlp_dim: int=2048,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.key_gate_scale = key_gate_scale
        proj_dim_1 = input_embed_dim // 2
        self.proj_1 = nn.Sequential(
            nn.Linear(input_embed_dim, proj_dim_1),
            nn.ReLU(),
            nn.Dropout(0.25)
        )
        self.cls_tokens = nn.Embedding(self.num_classes, proj_dim_1)
        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=proj_dim_1,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=nn.ReLU,
                    attention_downsample_rate=2,
                    skip_first_layer_pe=(i == 0),
                    use_self_attn = True
                )
            )
        self.cls_pos_heads = nn.ModuleList()
        for i in range(self.num_classes):
            self.cls_pos_heads.append(nn.Linear(proj_dim_1, 1))
    
    def forward(self, img_tokens, key_gate=None):
        bs, num_tokens, _ = img_tokens.shape
        if key_gate is not None:
            if key_gate.ndim == 1:
                key_gate = key_gate.unsqueeze(0)
            assert key_gate.shape == (bs, num_tokens), \
                f"MLC key_gate shape must be {(bs, num_tokens)}, got {tuple(key_gate.shape)}."
            key_gate = key_gate.to(device=img_tokens.device, dtype=img_tokens.dtype).detach()
            img_tokens = img_tokens * (1.0 + self.key_gate_scale * key_gate.unsqueeze(-1))
        keys = self.proj_1(img_tokens)  # (bs, num_tokens, C1=512)

        feat_size = int(math.sqrt(num_tokens))
        assert feat_size * feat_size == num_tokens, \
            f"MLCQuery expects square token maps, got {num_tokens} tokens."

        queries = self.cls_tokens.weight.unsqueeze(0).expand(bs, -1, -1)
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                # key_pe=key_pe,
                key_pe=None,
            )
        # queries: (bs, n_cls, dim), keys: (bs, num_tokens, dim)
        pred_pos_logits = []
        for i in range(self.num_classes):
            pred_pos_logits.append(self.cls_pos_heads[i](queries[:,i,:]))  # [(bs, 1),]
        pred_pos_logits = torch.cat(pred_pos_logits, dim=-1)  # (bs, n_cls)

        return pred_pos_logits,queries

class WSCerMLC(MetaClassifier):
    def __init__(self, args):
        evaluator = build_evaluator([ExtendMultiLabelMetric(
            thr = args.positive_thr,
            num_classes = args.num_classes,
            logger_name = args.logger_name,
            with_binary = True
        )])
        super(WSCerMLC, self).__init__(evaluator, args)
        input_embed_dim = args.backbone_cfg['backbone_token_output_dim'][-1]
        self.num_classes = args.num_classes
        self.format_heatmap = getattr(args, 'format_heatmap', False)
        self.format_img_token = getattr(args, 'format_img_token', False)
        self.binary_branch = CHIEF(input_embed_dim)
        self.mlc_branch = MLCQuery(args.num_classes, input_embed_dim, args.key_gate_scale, depth=2)
        self.use_pos_loss_weight = args.loss_cfg['type'] != 'AsymmetricLossOptimized'
        if self.use_pos_loss_weight:
            self.pos_loss_fn = build_loss(args.loss_cfg, reduction='none')
        else:
            self.pos_loss_fn = build_loss(args.loss_cfg)
        
    def calc_logits(self, inputs):
        img_tokens = self.get_img_tokens(inputs)  # (bs, num_tokens, C)
        pred_pn_logits, inter_var = self.binary_branch(img_tokens)

        key_gate = self.binary_branch.patch_probs(inter_var)['patch_prob']
        pred_pos_logits,pos_cls_tokens = self.mlc_branch(img_tokens, key_gate=key_gate)
        inter_var['pos_cls_tokens'] = pos_cls_tokens
        
        out = torch.cat([pred_pn_logits, pred_pos_logits], dim=-1)   # (bs, n_cls+1)
        return out, inter_var
    
    def calc_loss(self, inputs, databatch):
        pred_logits,_ = self.calc_logits(inputs)
        img_pn_logit = pred_logits[:, 0].unsqueeze(1)
        positive_logits = pred_logits[:, 1:]
        image_labels = torch.tensor([int(len(item.gt_label)>0) for item in databatch['data_samples']])
        img_gt = image_labels.to(self.device).unsqueeze(-1).float()
        pn_loss = F.binary_cross_entropy_with_logits(img_pn_logit, img_gt, reduction='mean')

        binary_matrix = self.get_mlc_labels(databatch)
        if self.use_pos_loss_weight:
            pos_loss_matrix = self.pos_loss_fn(positive_logits, binary_matrix)
            pos_loss_weight = torch.where(
                img_gt > 0,
                torch.ones_like(img_gt),
                torch.full_like(img_gt, 0.1),
            )
            pos_loss = (pos_loss_matrix * pos_loss_weight).mean()
        else:
            pos_loss = self.pos_loss_fn(positive_logits, binary_matrix)

        loss = pn_loss + pos_loss
        loss_dict = {
            'pn_loss': pn_loss.item(),
            'pos_loss': pos_loss.item(),
        }
        return loss,loss_dict

    def build_img_token(self, img_probs, pos_probs, inter_var):
        img_pn_feat = inter_var['img_feat']     # (bs, dim)
        pn_prob_feat = torch.cat([img_probs.unsqueeze(-1),img_pn_feat],dim=1)   # (bs, 1+dim)
        img_pos_feat = inter_var['pos_cls_tokens']  # (bs, n_cls, dim)
        pos_prob_feat = torch.cat([pos_probs.unsqueeze(-1),img_pos_feat],dim=2)  # (bs, n_cls, 1+dim)
        img_clstokens = torch.cat([pn_prob_feat.unsqueeze(1),pos_prob_feat],dim=1)     # (bs, 1+n_cls, 1+dim)

        return img_clstokens

    def set_pred(self, inputs, databatch):
        # attn_array: (bs, num_classes, num_tokens)
        pred_logits, inter_var = self.calc_logits(inputs) # (bs, num_classes)
        img_pn_logit = pred_logits[:, 0]
        positive_logits = pred_logits[:, 1:]
        img_probs = torch.sigmoid(img_pn_logit)   # (bs, )
        pos_probs = torch.sigmoid(positive_logits) # (bs, n_cls)

        if self.format_heatmap:
            bs_heatmap = (self.binary_branch.patch_probs(inter_var))['patch_prob']  # (bs, num_tokens)
        if self.format_img_token:
            bs_img_token = self.build_img_token(img_probs, pos_probs, inter_var)

        data_sampels = []
        for idx, (item, pn_p, pos_p) in enumerate(zip(databatch['data_samples'], img_probs, pos_probs)):
            item.img_prob = pn_p
            item.pos_prob = pos_p
            if self.format_heatmap:
                item.attn = bs_heatmap[idx]
            if self.format_img_token:
                item.img_token = bs_img_token[idx]
            data_sampels.append(item)

        return data_sampels
