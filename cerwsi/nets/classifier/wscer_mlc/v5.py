import torch
from torch import Tensor, nn
import math
import torch.nn.functional as F
from typing import Tuple, Type
from .feat_pe import get_feat_pe
from .chief import CHIEF
from ..meta_classifier import MetaClassifier
from cerwsi.utils import build_evaluator, ExtendMultiLabelMetric

class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))

class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
        use_self_attn: bool = True,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        self.use_self_attn = use_self_attn
        if use_self_attn:
            self.self_attn = Attention(embedding_dim, num_heads)
            self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self, queries: Tensor, keys: Tensor, key_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:
        
        # Self attention block
        if self.use_self_attn:
            if self.skip_first_layer_pe:
                queries,_ = self.self_attn(q=queries, k=queries, v=queries)
            else:
                attn_out,_ = self.self_attn(q=queries, k=queries, v=queries)
                queries = queries + attn_out
            queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries
        if key_pe is not None:
            k = keys + key_pe
        else:
            k = keys
        attn_out, attn_score = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries
        if key_pe is not None:
            k = keys + key_pe
        else:
            k = keys
        attn_out,_ = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)
        
        return queries, keys, attn_score

class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn_ = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn_, dim=-1)  # (bs, num_heads, num_cls, L)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out, attn_


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
            # self.cls_pos_heads.append(nn.Linear(num_patches, 1))
            self.cls_pos_heads.append(nn.Linear(proj_dim_1, 1))
    
    def forward(self, img_tokens):
        keys_1 = self.proj_1(img_tokens)  # (bs, num_tokens, C1=512)
        bs, num_tokens, embed_dim = keys_1.shape
        feat_size = int(math.sqrt(num_tokens))
        # key_pe: (1, embed_dim, feat_size[0], feat_size[1])
        key_pe = get_feat_pe('sam', embed_dim, (feat_size,feat_size))
        key_pe = key_pe.flatten(2).permute(0, 2, 1).to(keys_1.device) #  (1, num_tokens, dim)

        queries = self.cls_tokens.weight.unsqueeze(0).expand(bs, -1, -1)
        for layer in self.layers:
            queries, keys_1, attn_out_q = layer(
                queries=queries,
                keys=keys_1,
                key_pe=key_pe,
            )
        # queries: (bs, n_cls, dim), keys_1: (bs, num_tokens, dim)
        pred_pos_logits = []
        for i in range(self.num_classes):
            # pred_pos_logits.append(self.cls_pos_heads[i](attn_map[:,i,:]))  # [(bs, 1),]
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

