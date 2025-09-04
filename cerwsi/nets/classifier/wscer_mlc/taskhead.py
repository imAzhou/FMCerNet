import torch
from torch import nn
import math
import torch.nn.functional as F
from .feat_pe import get_feat_pe
from .chief import CHIEF
from .twoway_attn import TwoWayAttentionBlock
from ..meta_classifier import MetaClassifier
from cerwsi.utils import build_evaluator, ExtendMultiLabelMetric


class MLCQuery(nn.Module):
    def __init__(
        self,
        num_classes,
        input_embed_dim,
        depth: int=2,
        num_heads: int=8,
        mlp_dim: int=2048,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
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
    
    def forward(self, img_tokens):
        keys = self.proj_1(img_tokens)  # (bs, num_tokens, C1=512)
        bs, num_tokens, embed_dim = keys.shape
        feat_size = int(math.sqrt(num_tokens))
        # key_pe: (1, embed_dim, feat_size[0], feat_size[1])
        key_pe = get_feat_pe('sam', embed_dim, (feat_size,feat_size))
        key_pe = key_pe.flatten(2).permute(0, 2, 1).to(keys.device) #  (1, num_tokens, dim)

        queries = self.cls_tokens.weight.unsqueeze(0).expand(bs, -1, -1)
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                key_pe=key_pe,
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
        input_embed_dim = args.backbone_cfg['backbone_output_dim'][-1]
        self.num_classes = args.num_classes
        self.binary_branch = CHIEF(input_embed_dim)
        self.mlc_branch = MLCQuery(args.num_classes, input_embed_dim)
        
    def calc_logits(self, inputs):
        img_tokens = self.get_img_tokens(inputs)  # (bs, num_tokens, C)
        bs, num_tokens, embed_dim = img_tokens.shape
        feat_size = int(math.sqrt(num_tokens))
        # key_pe: (1, embed_dim, feat_size[0], feat_size[1])
        key_pe = get_feat_pe('sam', embed_dim, (feat_size,feat_size))
        key_pe = key_pe.flatten(2).permute(0, 2, 1).to(self.device) #  (1, num_tokens, dim)
        pred_pn_logits, inter_var = self.binary_branch(img_tokens+key_pe)

        pred_pos_logits,pos_cls_tokens = self.mlc_branch(img_tokens)
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

        loss_fn = nn.BCEWithLogitsLoss()
        binary_matrix = self.get_mlc_labels(databatch)
        pos_loss = loss_fn(positive_logits, binary_matrix)

        loss = pn_loss + pos_loss
        loss_dict = {
            'pn_loss': pn_loss.item(),
            'pos_loss': pos_loss.item(),
        }
        return loss,loss_dict

    def set_pred(self, inputs, databatch):
        # attn_array: (bs, num_classes, num_tokens)
        pred_logits, inter_var = self.calc_logits(inputs) # (bs, num_classes)
        img_pn_logit = pred_logits[:, 0]
        positive_logits = pred_logits[:, 1:]
        img_probs = torch.sigmoid(img_pn_logit).squeeze(-1)   # (bs, )
        bs = len(databatch['data_samples'])
        if bs == 1:
            img_probs = img_probs.unsqueeze(0)
        pos_probs = torch.sigmoid(positive_logits) # (bs, n_cls)
        
        heatmap = (self.binary_branch.patch_probs(inter_var))['patch_prob']  # (bs, num_tokens)
        _,clsid = torch.max(pos_probs, 1)       # (bs, )
        img_pn_feat = inter_var['img_feat']     # (bs, dim)
        img_pos_feat = inter_var['pos_cls_tokens'][torch.arange(bs), clsid]     # (bs, dim)
        img_clstokens = torch.cat([img_pn_feat,img_pos_feat],dim=1)     # (bs, dim*2)

        data_sampels = []
        for item, pn_p, pos_p, imgtoken, attn in zip(databatch['data_samples'], img_probs, pos_probs, img_clstokens, heatmap):
            item.img_prob = pn_p
            item.pos_prob = pos_p
            item.attn = attn  # (num_tokens)
            item.img_token = imgtoken
            data_sampels.append(item)

        return data_sampels

