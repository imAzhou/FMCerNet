import torch
import torch.nn as nn
import math
from fmcernet.utils import build_evaluator, SlideMetric
from mmengine.optim import OptimWrapper
from .get_mil import get_mil

class TileTokenAggregator(nn.Module):
    def __init__(self, feat_dim=512, num_pos_tokens=5, prob_eps=1e-6):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_pos_tokens = num_pos_tokens
        self.token_dim = feat_dim + 1
        self.raw_dim = (num_pos_tokens + 1) * self.token_dim
        self.prob_eps = prob_eps
        self.query = nn.Parameter(torch.empty(feat_dim))
        nn.init.normal_(self.query, std=feat_dim ** -0.5)

    def forward(self, x):
        if x.dim() != 3 or x.size(-1) != self.raw_dim:
            raise ValueError(f'TileTokenAggregator expects input shape (B, K, {self.raw_dim}), got {tuple(x.shape)}.')

        b, k, _ = x.shape
        tokens = x.view(b, k, self.num_pos_tokens + 1, self.token_dim)
        pn_feat = tokens[:, :, 0, 1:]
        pos_prob = tokens[:, :, 1:, 0]
        pos_feat = tokens[:, :, 1:, 1:]

        attn_logits = (pos_feat * self.query.view(1, 1, 1, -1)).sum(dim=-1)
        attn_logits = attn_logits / math.sqrt(self.feat_dim)
        attn_logits = attn_logits + torch.log(pos_prob.clamp_min(self.prob_eps))
        attn = torch.softmax(attn_logits, dim=-1)
        pos_feat_agg = (attn.unsqueeze(-1) * pos_feat).sum(dim=2)
        return torch.cat([pn_feat, pos_feat_agg], dim=-1)

class SlideNet(nn.Module):
    def __init__(self, cfg):
        super(SlideNet, self).__init__()
        self.tile_token_aggregator = None
        if cfg.get('tile_token_aggregator', False):
            self.tile_token_aggregator = TileTokenAggregator()
        self.mil_model = get_mil(cfg)
        self.taskhead = nn.Identity()
        self.taskhead.evaluator = build_evaluator([SlideMetric(
            num_classes = cfg.num_classes,
            logger_name = cfg.logger_name
        )])
        
    @property
    def device(self):
        return next(self.parameters()).device

    def load_ckpt(self, ckpt):
        params_weight = torch.load(ckpt, map_location=self.device)
        print(self.load_state_dict(params_weight, strict=False))
    
    def forward(self, data_batch, mode, optim_wrapper=None):
        if mode == 'train':
            return self.train_step(data_batch, optim_wrapper)
        if mode == 'val':
            return self.val_step(data_batch)

    def aggregate_tile_tokens(self, databatch):
        if self.tile_token_aggregator is None:
            return databatch
        agg_databatch = dict(databatch)
        agg_databatch['inputs'] = self.tile_token_aggregator(databatch['inputs'].to(self.device))
        return agg_databatch
    
    def train_step(self, databatch, optim_wrapper: OptimWrapper):
        databatch = self.aggregate_tile_tokens(databatch)
        loss,loss_dict = self.mil_model.calc_loss(databatch)
        optim_wrapper.update_params(loss)
        return loss,loss_dict

    def val_step(self, databatch):
        databatch = self.aggregate_tile_tokens(databatch)
        loss, loss_dict = self.mil_model.calc_loss(databatch)
        databatch = self.mil_model.set_pred(databatch)
        return databatch, loss, loss_dict
