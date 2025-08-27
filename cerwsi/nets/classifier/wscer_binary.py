import torch
from torch import Tensor, nn
import math
import torch.nn.functional as F
from typing import Tuple, Type
from .wscer_mlc.feat_pe import get_feat_pe
from .meta_classifier import MetaClassifier
from cerwsi.utils import build_evaluator,BinaryMetric

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
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
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
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)
        
        return queries, keys

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

        return out


class WSCerBinary(MetaClassifier):
    def __init__(self, args):
        input_dim = args.neck_output_dim[0]
        evaluator = build_evaluator([BinaryMetric(thr = args.positive_thr)])
        super(WSCerBinary, self).__init__(evaluator, **args)

        depth = 2
        num_heads = 8
        mlp_dim = 2048
        use_self_attn = False
        self.pos_add_type = 'sam' # 'sam','query2label',None
        self.num_classes = 1
        
        self.cls_tokens = nn.Embedding(1, input_dim)
        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=input_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=nn.ReLU,
                    attention_downsample_rate=2,
                    skip_first_layer_pe=(i == 0),
                    use_self_attn = use_self_attn
                )
            )
        self.cls_neg_head = nn.Linear(input_dim, 1)
        

    def calc_logits(self, img_tokens: torch.Tensor):
        bs, num_tokens, embed_dim = img_tokens.shape
        feat_size = int(math.sqrt(num_tokens))
        queries = self.cls_tokens.weight.unsqueeze(0).expand(bs, -1, -1)
        key_pe = None
        if self.pos_add_type is not None:
            # key_pe: (1, embed_dim, feat_size[0], feat_size[1])
            key_pe = get_feat_pe(self.pos_add_type, embed_dim, (feat_size,feat_size))
            key_pe = key_pe.flatten(2).permute(0, 2, 1).to(self.device)

        for layer in self.layers:
            queries, img_tokens = layer(
                queries=queries,
                keys=img_tokens,
                key_pe=key_pe,
            )

        cls_pn_token = queries[:,0,:]  # (bs, C)
        pred_pn_logits = self.cls_neg_head(cls_pn_token)  # (bs, 1)
        
        return pred_pn_logits
    
    def calc_pos_loss(self, pos_logits, databatch):
        loss_fn = nn.BCEWithLogitsLoss()
        binary_matrix = databatch['multi_pos_labels'].to(self.device)
        loss = loss_fn(pos_logits, binary_matrix)
        return loss
    
    def calc_loss(self,feature_emb, databatch):
        img_pn_logit = self.calc_logits(feature_emb)
        img_gt = databatch['image_labels'].to(self.device).unsqueeze(-1).float()
        loss = F.binary_cross_entropy_with_logits(img_pn_logit, img_gt, reduction='mean')
        return loss

    def set_pred(self,feature_emb, databatch):
        img_pn_logit = self.calc_logits(feature_emb) # (bs, num_classes)
        databatch['img_probs'] = torch.sigmoid(img_pn_logit).squeeze(-1)   # (bs, )
        databatch['attn_array'] = None
        return databatch
