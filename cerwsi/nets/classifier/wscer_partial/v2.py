import torch
from torch import Tensor, nn
import math
import torch.nn.functional as F
from typing import List, Tuple, Type
from torchvision.ops import nms
import numpy as np
import cv2
from ..wscer_mlc.feat_pe import get_feat_pe
from ..meta_classifier import MetaClassifier
from cerwsi.nets.backbone.SAM.common import LayerNorm2d,MLP
from cerwsi.utils import build_evaluator,ImgODMetric

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

class TokenClsBranch(nn.Module):
    def __init__(self, num_classes,input_embed_dim):
        super(TokenClsBranch, self).__init__()
        depth = 2
        num_heads = 8
        mlp_dim = 2048
        self.pos_add_type = 'sam' # 'sam','query2label',None
        self.num_classes = num_classes

        self.mask_fc = nn.Linear(input_embed_dim, 1)

        self.cls_tokens = nn.Embedding(num_classes, input_embed_dim)
        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=input_embed_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=nn.ReLU,
                    attention_downsample_rate=2,
                    skip_first_layer_pe=(i == 0),
                )
            )
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(input_embed_dim, input_embed_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(input_embed_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(input_embed_dim // 4, input_embed_dim // 8, kernel_size=2, stride=2),
            nn.GELU(),
            nn.ConvTranspose2d(input_embed_dim // 8, input_embed_dim // 16, kernel_size=2, stride=2),
            nn.GELU()
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(input_embed_dim, input_embed_dim, input_embed_dim // 16, 3)
                for i in range(self.num_classes)
            ]
        )
    
    def forward(self, img_tokens: torch.Tensor, device):
        bs, num_tokens, embed_dim = img_tokens.shape
        feat_size = int(math.sqrt(num_tokens))
        queries = self.cls_tokens.weight.unsqueeze(0).expand(bs, -1, -1)
        key_pe = None
        if self.pos_add_type is not None:
            # key_pe: (1, embed_dim, feat_size[0], feat_size[1])
            key_pe = get_feat_pe(self.pos_add_type, embed_dim, (feat_size,feat_size))
            key_pe = key_pe.flatten(2).permute(0, 2, 1).to(device)

        mask_logits = self.mask_fc(img_tokens)  # (bs, num_tokens, 1)

        attn_array = []
        for layer in self.layers:
            queries, img_tokens, attn_out_q = layer(
                queries=queries,
                keys=img_tokens,
                key_pe=key_pe,
            )
            # attn_out_q: (bs, num_heads, num_cls, L)
            # attn_score: (bs, num_cls, L)
            attn_score = torch.mean(attn_out_q, dim=1)
            attn_array.append(attn_score)
        
        img_tokens = img_tokens.transpose(1, 2).view(bs, embed_dim, feat_size, feat_size)
        upscaled_embedding = self.output_upscaling(img_tokens)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_classes):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](queries[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        attn_map = (hyper_in @ upscaled_embedding.view(b, c, h * w))   # (bs, n_cls, num_tokens)
        
        # attn_array = torch.stack(attn_array, dim=1)
        # return mask_logits, attn_map, attn_array
        return mask_logits, attn_map
    
class BinaryClsBranch(nn.Module):
    def __init__(self, embed_dim):
        super(BinaryClsBranch, self).__init__()

        self.downsample = nn.Sequential(
            nn.Conv2d(embed_dim+1, 1024, kernel_size=3, stride=2, padding=1),  # 下采样 x2
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 2048, kernel_size=3, stride=2, padding=1),  # 再次下采样 x2，总共下采样 x4
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)  # 输出大小为 [bs, 2048, 1, 1]
        self.fc = nn.Linear(2048, 1)


    def forward(self, img_tokens, mask):
        bs, num_tokens, embed_dim = img_tokens.shape
        H = W = int(math.sqrt(num_tokens))
        x = img_tokens.transpose(1, 2).reshape(bs, embed_dim, H, W)

        # concat mask: [bs, embed_dim + 1, H, W]
        x = torch.cat([x, mask], dim=1)
        x = self.downsample(x)  # [bs, 2048, H/4, W/4]
        x = self.avgpool(x)     # [bs, 2048, 1, 1]
        x = x.view(bs, -1)      # [bs, 2048]
        logit = self.fc(x)      # [bs, 1]

        return logit

def apply_cutout_mask(
        map_size,
        hole_height_range=(1, 6),
        hole_width_range=(1, 6),
        num_holes_range=(0, 5)
    ):
    """
    在特征图上应用 Cutout 掩码，遮挡若干随机大小的矩形区域。

    Args:
        map_size: tuple, [B, C, H, W]
        hole_height_range: (min_h, max_h)
        hole_width_range: (min_w, max_w)
        num_holes_range: (min_n, max_n)
    Returns:
        mask: Tensor, [B, 1, H, W]，其中 0 表示被遮挡区域
    """
    B, C, H, W = map_size
    mask = torch.ones(map_size)
    for b in range(B):
        num_holes = torch.randint(num_holes_range[0], num_holes_range[1] + 1, (1,)).item()
        for _ in range(num_holes):
            hole_h = torch.randint(hole_height_range[0], hole_height_range[1] + 1, (1,)).item()
            hole_w = torch.randint(hole_width_range[0], hole_width_range[1] + 1, (1,)).item()

            # 随机位置，确保不会越界
            y1 = torch.randint(0, max(H - hole_h + 1, 1), (1,)).item()
            x1 = torch.randint(0, max(W - hole_w + 1, 1), (1,)).item()
            mask[b, 0, y1:y1 + hole_h, x1:x1 + hole_w] = 0
    return mask

class WSCerPartial(MetaClassifier):
    def __init__(self, args):
        input_embed_dim = args.neck_output_dim[0]
        num_classes = args.num_classes
        save_result_dir = getattr(args, 'save_result_dir', None)
        evaluator = build_evaluator([ImgODMetric(args.logger_name,save_result_dir)])
        super(WSCerPartial, self).__init__(evaluator, **args)

        self.token_cls_branch = TokenClsBranch(num_classes, input_embed_dim)
        self.binary_cls_branch = BinaryClsBranch(input_embed_dim)


    def calc_logits(self, img_tokens: torch.Tensor, apply_auxiliary: str|None):
        '''
        mask_logits: (bs, num_tokens, 1), num_tokens 数量与img_tokens一致
        token_logits: (bs, n_cls, num_tokens) num_tokens upsampled k ratio
        img_logits: (bs, 1), the logits of img belong to pos/neg
        masks: (bs, 1, H, W) img_tokens 的 tokens 数量等于 H*W
        '''
        bs, num_tokens, embed_dim = img_tokens.shape
        H = W = int(math.sqrt(num_tokens))
        mask_logits, token_logits = self.token_cls_branch(img_tokens, self.device)
        mask_logits = mask_logits.squeeze(-1).view(bs, H, W)

        bs, n_cls, upscaled_tokens = token_logits.shape
        UH = UW = int(math.sqrt(upscaled_tokens))
        token_logits = token_logits.view(bs, n_cls, UH, UW)
        
        auxiliary_masks = torch.ones((bs,1,H,W)).to(img_tokens.device)
        if apply_auxiliary == 'random':
            auxiliary_masks = (apply_cutout_mask((bs,1,H,W))).to(img_tokens.device)
        elif apply_auxiliary == 'logit':
            auxiliary_masks = torch.relu(mask_logits).unsqueeze(1)   # (bs,1,H,W)
        img_logits = self.binary_cls_branch(img_tokens, auxiliary_masks)
        auxiliary_masks = (auxiliary_masks > 0.5).int()
        return mask_logits, token_logits, img_logits, auxiliary_masks
     
    def create_feat_gt(self, auxiliary_masks, databatch, device):
        bs, _, H, W = auxiliary_masks.shape

        # Step 1: Resize clsid_mask to (H, W) using nearest interpolation
        clsid_mask = databatch['clsid_mask'].float().unsqueeze(1).to(device)  # (bs, 1, H_orig, W_orig)
        clsid_mask_HW = F.interpolate(clsid_mask, size=(H, W), mode='nearest')     # (bs, 1, H, W)
        mask_gt = (clsid_mask_HW > 1).float()  # binary mask: >1 → 1, else 0

        # Step 2: Compute img_gt
        aux_mul_mask = (mask_gt * auxiliary_masks).sum(dim=(1, 2, 3))  # (bs,)
        img_gt = torch.full((bs,), -1, device=device)  # init as -1

        for i in range(bs):
            if aux_mul_mask[i] > 0:
                img_gt[i] = 1
            else:
                zero_aux_mask = (auxiliary_masks[i, 0] == 0)
                clsid_vals = clsid_mask_HW[i,0][zero_aux_mask]
                if (clsid_vals == 1).all():
                    img_gt[i] = 0

        # Step 3: Resize clsid_mask to (H*4, W*4) and subtract 1 at nonzero positions
        clsid_mask_HW4 = F.interpolate(clsid_mask, size=(H * 8, W * 8), mode='nearest').long().squeeze(1)  # (bs, H*4, W*4)
        
        # Step 4: Build balanced_mask dict
        balanced_mask = {
            'img_ignore': img_gt == -1,                                # (bs,)
            'mask_ignore': clsid_mask_HW == 0,                         # (bs, H, W)
            'token_ignore': clsid_mask_HW4 == 0                        # (bs, H*4, W*4)
        }

        clsid_mask_HW4[clsid_mask_HW4 != 0] -= 1  # so value range becomes [0, 5]
        return mask_gt.squeeze(1), clsid_mask_HW4, img_gt, balanced_mask

    def calc_loss(self, feature_emb, databatch, epoch):
        loss_fn_1 = nn.BCEWithLogitsLoss(reduction='none')
        loss_fn_2 = nn.CrossEntropyLoss(reduction='none')
        loss_fn_3 = nn.BCEWithLogitsLoss(reduction='none')

        # 计算logits与gt
        apply_auxiliary = 'random' if epoch < 20 else 'logit'
        # apply_auxiliary = 'logit'
        mask_logits, token_logits, img_logits, auxiliary_masks = self.calc_logits(feature_emb, apply_auxiliary)
        mask_gt, token_gt, img_gt, balanced_mask = self.create_feat_gt(auxiliary_masks, databatch, self.device)

        # mask_loss
        mask_loss = loss_fn_1(mask_logits, mask_gt)  # (bs, H, W)
        mask_loss = mask_loss.masked_fill(balanced_mask['mask_ignore'], 0.0)
        valid_mask = (~balanced_mask['mask_ignore']).float()
        mask_loss = (mask_loss * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)

        # token_loss
        token_loss = loss_fn_2(token_logits, token_gt)  # (bs, H4, W4)
        token_loss = token_loss.masked_fill(balanced_mask['token_ignore'], 0.0)
        valid_token = (~balanced_mask['token_ignore']).float()
        token_loss = (token_loss * valid_token).sum() / valid_token.sum().clamp(min=1.0)

        # img_loss
        img_logits = img_logits.view(-1)
        img_gt = img_gt.view(-1).float()
        img_valid_mask = ~balanced_mask['img_ignore']  # (bs,)
        img_loss = loss_fn_3(img_logits, img_gt)
        img_loss = img_loss * img_valid_mask.float()
        img_loss = img_loss.sum() / img_valid_mask.sum().clamp(min=1.0)

        # 总损失
        loss = mask_loss + token_loss + img_loss
        loss_dict = {
            'mask_loss': mask_loss.item(),
            'token_loss': token_loss.item(),
            'img_loss': img_loss.item(),
        }
        return loss, loss_dict

    def set_pred(self,feature_emb, databatch):
        mask_logits, token_logits, img_logits, auxiliary_masks = self.calc_logits(feature_emb, None)
        token_cls_pred = F.softmax(token_logits, dim=1)  # (bs, n_cls, H4, W4)
        databatch['img_probs'] = torch.sigmoid(img_logits).squeeze(-1)   # (bs, )

        input_x = databatch['images']   # (bs, c, h, w)
        imgh,imgw = input_x.shape[2:]
        databatch['pred_bbox'] = self.post_process(token_cls_pred, (imgh,imgw))
        return databatch

    def post_process(self, token_cls_pred, input_size, min_box_size=25, iou_thresh=0.7):
        '''
        Args:
            token_cls_pred: (bs, n_cls, H4, W4) softmax后的token级别分类概率
        Returns:
            pred_bboxes: List[List[Dict]], 每张图对应一个预测框列表，每个框格式如下：
                            {'bbox': [x1, y1, x2, y2], 'score': float, 'cls': int}
        '''
        bs, n_cls, H, W = token_cls_pred.shape
        pred_bboxes_batch = []

        for b in range(bs):
            # token_map: (n_cls, H, W)
            token_map = token_cls_pred[b].detach().cpu().numpy()
            token_score = np.max(token_map, axis=0)         # (H, W)
            token_class = np.argmax(token_map, axis=0)      # (H, W)

            bboxes = []
            for cls_id in range(1, n_cls):  # 忽略 cls==0（阴性或未知）
                cls_mask = (token_class == cls_id).astype(np.uint8)  # (H, W)

                # 连通域提取（OpenCV）
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cls_mask, connectivity=8)

                for i in range(1, num_labels):  # 0是背景
                    x, y, w, h, area = stats[i]
                    x1, y1, x2, y2 = x, y, x + w, y + h

                    # 小于阈值则扩展到 min_box_size
                    if w < min_box_size or h < min_box_size:
                        cx, cy = centroids[i]
                        cx, cy = int(cx), int(cy)
                        half = min_box_size // 2
                        x1 = max(0, cx - half)
                        y1 = max(0, cy - half)
                        x2 = min(W, cx + half)
                        y2 = min(H, cy + half)

                    # 获取这个连通域内的最大 score
                    mask = (labels == i)
                    score = token_score[mask].max()

                    bboxes.append({
                        'bbox': [x1, y1, x2, y2],
                        'score': float(score),
                        'cls': cls_id
                    })

            # NMS per class
            final_boxes = []
            for cls_id in range(1, n_cls):
                cls_boxes = [box for box in bboxes if box['cls'] == cls_id]
                if not cls_boxes:
                    continue

                boxes_tensor = torch.tensor([box['bbox'] for box in cls_boxes], dtype=torch.float32)
                scores_tensor = torch.tensor([box['score'] for box in cls_boxes], dtype=torch.float32)
                keep = nms(boxes_tensor, scores_tensor, iou_thresh)

                for idx in keep:
                    final_boxes.append(cls_boxes[idx])

            # 将 H,W 的 bbox 尺度映射回原图尺寸
            orig_h, orig_w = input_size
            scale_x = orig_w / W
            scale_y = orig_h / H
            for box in final_boxes:
                x1, y1, x2, y2 = box['bbox']
                box['bbox'] = [
                    int(x1 * scale_x),
                    int(y1 * scale_y),
                    int(x2 * scale_x),
                    int(y2 * scale_y)
                ]

            pred_bboxes_batch.append(final_boxes)

        return pred_bboxes_batch
    