# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Union, Tuple, Type
from torch import Tensor
import torch
import torch.nn.functional as F
from torchvision.ops import nms
from torch import nn
from mmdet.models.task_modules import HungarianAssigner,MaskPseudoSampler
from mmdet.models.losses import CrossEntropyLoss,DiceLoss
from mmdet.utils import InstanceList,reduce_mean
from mmdet.models.utils import get_uncertain_point_coords_with_randomness,multi_apply
from mmcv.ops import point_sample
from mmengine.structures import InstanceData
from .transformer import TwoWayTransformer
from sam2.modeling.sam2_utils import LayerNorm2d, MLP
from sam2.modeling.sam.prompt_encoder import PromptEncoder


class Instance_branch(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        img_input_size: int,
        num_instance_queries: int,
        num_classes: int,   # 0:阴性，>0:阳性类别
        activation: Type[nn.Module] = nn.GELU,
        pretrain_ckpt = None,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = TwoWayTransformer(
            depth=2,
            embedding_dim=transformer_dim,
            mlp_dim=2048,
            num_heads=8,
        )
        self.instance_queries = num_instance_queries
        self.num_classes = num_classes
        self.mask_tokens = nn.Embedding(self.instance_queries, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(
                transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
            ),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(
                transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
            ),
            activation(),
        )
        self.backbone_stride = 16
        self.img_input_size = img_input_size
        self.sam_image_embedding_size = self.img_input_size // self.backbone_stride
        self.sam_prompt_encoder = PromptEncoder(
            embed_dim=transformer_dim,
            image_embedding_size=(
                self.sam_image_embedding_size,
                self.sam_image_embedding_size,
            ),
            input_image_size=(self.img_input_size, self.img_input_size),
            mask_in_chans=16,
        )
        for name, param in self.sam_prompt_encoder.named_parameters():
            param.requires_grad = False

        self.conv_s0 = nn.Conv2d(transformer_dim, transformer_dim // 8, kernel_size=1, stride=1)
        self.conv_s1 = nn.Conv2d(transformer_dim, transformer_dim // 4, kernel_size=1, stride=1)
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.instance_queries)
            ]
        )

        self.cls_prediction_head = MLP(transformer_dim, 256, num_classes, 3)
        # a single token to indicate no memory embedding from previous frames
        self.no_mem_embed = torch.nn.Parameter(torch.zeros(1, 1, transformer_dim))

        self.assigner = HungarianAssigner(match_costs=[
                dict(type='ClassificationCost', weight=2.0),
                dict(type='CrossEntropyLossCost', weight=5.0, use_sigmoid=True),
                dict(type='DiceCost', weight=5.0, pred_act=True, eps=1.0)
            ])
        self.sampler = MaskPseudoSampler()
        self.num_points = 112*112
        self.oversample_ratio = 3.0
        self.importance_sample_ratio = 0.75

        self.class_weight = [0.1] + [1.0] * (num_classes-1)
        self.loss_cls = CrossEntropyLoss(class_weight=self.class_weight)
        self.loss_mask = CrossEntropyLoss(use_sigmoid=True)
        self.loss_dice = DiceLoss(naive_dice=True, eps=1.0)

        if pretrain_ckpt is not None:
            self.load_ckpt(pretrain_ckpt)
        
    def load_ckpt(self, pretrain_ckpt):
        """load ckpt from sam2"""
        params_weight = torch.load(pretrain_ckpt, map_location="cpu", weights_only=True)["model"]
        retain_keys = [
            'transformer', 'output_upscaling', 'conv_s0', 'conv_s1'
        ]
        state_dict = {}
        for key,value in params_weight.items():
            if key == 'no_mem_embed':
                state_dict[key] = value
            if 'sam_mask_decoder' in key:
                new_name = key.replace('sam_mask_decoder.', '')
                first_tag = new_name.split('.')[0]
                if first_tag in retain_keys:
                    state_dict[new_name] = value
        load_result = self.load_state_dict(state_dict, strict=False)
        print('Load decoder from SAM2: ' + str(load_result))

    def forward(self, dict_inputs, mask_input=None):
        """
        mask_input: can be logits which shape is (bs, 1, H, W) where for SAM, H=W=image_embed HW.

        Returns:
            masks: (bs, num_instance_queries, h, w)
            cls_pred:  (bs, num_instance_queries, num_class)
        """
        # Add no_mem_embed, which is added to the lowest rest feat. map
        batch_size, lowest_c, lowest_h, lowest_w = dict_inputs['vision_features'].shape
        image_embed = dict_inputs['vision_features'].flatten(2).permute(2, 0, 1)    # flatten NxCxHxW to HWxNxC
        image_embed = image_embed + self.no_mem_embed
        image_embed = image_embed.permute(1, 2, 0).view(batch_size, -1, lowest_h, lowest_w)    # NxCxHxW
        
        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(
            points=None,
            boxes=None,
            masks=mask_input,
        )
        image_pe = self.sam_prompt_encoder.get_dense_pe()

        # Concatenate output tokens
        output_tokens = self.mask_tokens.weight.unsqueeze(0).expand(batch_size, -1, -1)
        src = image_embed + dense_embeddings

        assert (
            image_pe.size(0) == 1
        ), "image_pe should have size 1 in batch dim (from `get_dense_pe()`)"
        pos_src = torch.repeat_interleave(image_pe, output_tokens.shape[0], dim=0)
        b, c, h, w = src.shape
        # Run the transformer
        mask_tokens_out, src = self.transformer(src, pos_src, output_tokens)
        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)

        # get high_res_features
        feat_s0 = self.conv_s0(dict_inputs['backbone_fpn'][0])    # NxCxHxW
        feat_s1 = self.conv_s1(dict_inputs['backbone_fpn'][1])    # NxCxHxW
        dc1, ln1, act1, dc2, act2 = self.output_upscaling
        upscaled_embedding = act1(ln1(dc1(src) + feat_s1))
        upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0)

        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.instance_queries):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            )
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask class predictions
        cls_pred = self.cls_prediction_head(mask_tokens_out)    # (bs, num_queries, num_class)

        return masks, cls_pred

    def loss(self, dict_inputs: dict, databatch, mask_input=None):
        '''
        dict_inputs: dict, 
            vision_features: Tensor, (bs, c, h, w)
            vision_pos_enc: List[Tensor]: [bs, c, h1,w1]...
            backbone_fpn: List[Tensor]: [bs, c, h1,w1]...
        '''
        pred_mask_logits, pred_cls_logits = self(dict_inputs, mask_input)
        num_imgs, _, logit_h, logit_w = pred_mask_logits.shape

        device = pred_cls_logits.device
        batch_gt_instances,cls_scores_list,mask_preds_list = [],[],[]
        for i in range(num_imgs):
            cls_scores_list.append(pred_cls_logits[i])
            mask_preds_list.append(pred_mask_logits[i])
            gt_instances = InstanceData()
            # img_label = databatch['image_labels'][i]
            if len(databatch['instance_mask'][i]) == 0:
                gt_instances.masks = torch.empty((0, logit_h, logit_w), device=device).float()
                gt_instances.labels = torch.empty((0,), device=device).long()
            else:
                instance_mask = F.interpolate(
                    torch.as_tensor(databatch['instance_mask'][i]).unsqueeze(1), 
                    size=(logit_h, logit_w), mode='nearest').squeeze(1)  # (1, H, W)
                gt_instances.masks = instance_mask.to(device).float()
                gt_instances.labels = torch.as_tensor(databatch['instance_label'][i], device=device).long()

            batch_gt_instances.append(gt_instances)
        
        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         avg_factor) = self.get_targets(cls_scores_list, mask_preds_list, batch_gt_instances, databatch['metainfo'])
        # shape (batch_size, num_queries)
        labels = torch.stack(labels_list, dim=0)
        # shape (batch_size, num_queries)
        label_weights = torch.stack(label_weights_list, dim=0)
        # shape (num_total_gts, h, w)
        mask_targets = torch.cat(mask_targets_list, dim=0)
        # shape (batch_size, num_queries)
        mask_weights = torch.stack(mask_weights_list, dim=0)

        # classfication loss
        # shape (batch_size * num_queries, )
        cls_scores = pred_cls_logits.flatten(0, 1)
        labels = labels.flatten(0, 1)
        label_weights = label_weights.flatten(0, 1)

        class_weight = cls_scores.new_tensor(self.class_weight)
        loss_cls = self.loss_cls(
            cls_scores,
            labels,
            label_weights,
            avg_factor=class_weight[labels].sum())

        num_total_masks = reduce_mean(cls_scores.new_tensor([avg_factor]))
        num_total_masks = max(num_total_masks, 1)

        # extract positive ones
        # shape (batch_size, num_queries, h, w) -> (num_total_gts, h, w)
        mask_preds = pred_mask_logits[mask_weights > 0]

        if mask_targets.shape[0] == 0:
            # zero match
            loss_dice = mask_preds.sum()
            loss_mask = mask_preds.sum()
            instance_loss_dict = {
                'loss_cls': loss_cls,
                'loss_mask': loss_mask,
                'loss_dice': loss_dice,
            }
            return instance_loss_dict

        with torch.no_grad():
            points_coords = get_uncertain_point_coords_with_randomness(
                mask_preds.unsqueeze(1), None, self.num_points,
                self.oversample_ratio, self.importance_sample_ratio)
            # shape (num_total_gts, h, w) -> (num_total_gts, num_points)
            mask_point_targets = point_sample(
                mask_targets.unsqueeze(1).float(), points_coords).squeeze(1)
        # shape (num_queries, h, w) -> (num_queries, num_points)
        mask_point_preds = point_sample(
            mask_preds.unsqueeze(1), points_coords).squeeze(1)

        # dice loss
        loss_dice = self.loss_dice(
            mask_point_preds, mask_point_targets, avg_factor=num_total_masks)

        # mask loss
        # shape (num_queries, num_points) -> (num_queries * num_points, )
        mask_point_preds = mask_point_preds.reshape(-1)
        # shape (num_total_gts, num_points) -> (num_total_gts * num_points, )
        mask_point_targets = mask_point_targets.reshape(-1)
        loss_mask = self.loss_mask(
            mask_point_preds,
            mask_point_targets,
            avg_factor=num_total_masks * self.num_points)

        instance_loss_dict = {
            'loss_cls': loss_cls,
            'loss_mask': loss_mask[0],
            'loss_dice': loss_dice[0],
        }
        return instance_loss_dict

    def get_targets(
        self,
        cls_scores_list: List[Tensor],
        mask_preds_list: List[Tensor],
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict]
    ) -> Tuple[List[Union[Tensor, int]]]:
        """
        Args:
            cls_scores_list (list[Tensor]): Mask score logits from a single
                decoder layer for all images. Each with shape (num_queries,
                cls_out_channels).
            mask_preds_list (list[Tensor]): Mask logits from a single decoder
                layer for all images. Each with shape (num_queries, h, w).
            batch_gt_instances (list[obj:`InstanceData`]): each contains
                ``labels`` and ``masks``.
            batch_img_metas (list[dict]): List of image meta information.
        Returns:
            tuple: a tuple containing the following targets.

                - labels_list (list[Tensor]): Labels of all images.\
                    Each with shape (num_queries, ).
                - label_weights_list (list[Tensor]): Label weights\
                    of all images. Each with shape (num_queries, ).
                - mask_targets_list (list[Tensor]): Mask targets of\
                    all images. Each with shape (num_queries, h, w).
                - mask_weights_list (list[Tensor]): Mask weights of\
                    all images. Each with shape (num_queries, ).
                - avg_factor (int): Average factor that is used to average\
                    the loss. When using sampling method, avg_factor is
                    usually the sum of positive and negative priors. When
                    using `MaskPseudoSampler`, `avg_factor` is usually equal
                    to the number of positive priors.

            additional_returns: This function enables user-defined returns from
                `self._get_targets_single`. These returns are currently refined
                to properties at each feature map (i.e. having HxW dimension).
                The results will be concatenated after the end.
        """
        results = multi_apply(self._get_targets_single, cls_scores_list,
                              mask_preds_list, batch_gt_instances, batch_img_metas)
        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         pos_inds_list, neg_inds_list, sampling_results_list) = results[:7]
        rest_results = list(results[7:])

        avg_factor = sum(
            [results.avg_factor for results in sampling_results_list])

        res = (labels_list, label_weights_list, mask_targets_list,
               mask_weights_list, avg_factor)

        return res + tuple(rest_results)

    def _get_targets_single(self, cls_score: Tensor, mask_pred: Tensor,
                            gt_instances: InstanceData, img_meta) -> Tuple[Tensor]:
        """Compute classification and mask targets for one image.

        Args:
            cls_score (Tensor): Mask score logits from a single decoder layer
                for one image. Shape (num_queries, cls_out_channels).
            mask_pred (Tensor): Mask logits for a single decoder layer for one
                image. Shape (num_queries, h, w).
            gt_instances (:obj:`InstanceData`): It contains ``labels`` and
                ``masks``.
            img_meta (dict): Image informtation.

        Returns:
            tuple[Tensor]: A tuple containing the following for one image.

                - labels (Tensor): Labels of each image. \
                    shape (num_queries, ).
                - label_weights (Tensor): Label weights of each image. \
                    shape (num_queries, ).
                - mask_targets (Tensor): Mask targets of each image. \
                    shape (num_queries, h, w).
                - mask_weights (Tensor): Mask weights of each image. \
                    shape (num_queries, ).
                - pos_inds (Tensor): Sampled positive indices for each \
                    image.
                - neg_inds (Tensor): Sampled negative indices for each \
                    image.
                - sampling_result (:obj:`SamplingResult`): Sampling results.
        """
        gt_labels = gt_instances.labels
        gt_masks = gt_instances.masks
        # sample points
        num_queries = cls_score.shape[0]
        num_gts = gt_labels.shape[0]

        point_coords = torch.rand((1, self.num_points, 2),
                                  device=cls_score.device)
        # shape (num_queries, num_points)
        mask_points_pred = point_sample(
            mask_pred.unsqueeze(1), point_coords.repeat(num_queries, 1, 1)).squeeze(1)
        # shape (num_gts, num_points)
        gt_points_masks = point_sample(
            gt_masks.unsqueeze(1).float(), point_coords.repeat(num_gts, 1,1)).squeeze(1)

        sampled_gt_instances = InstanceData(
            labels=gt_labels, masks=gt_points_masks)
        sampled_pred_instances = InstanceData(
            scores=cls_score, masks=mask_points_pred)
        # assign and sample
        assign_result = self.assigner.assign(
            pred_instances=sampled_pred_instances,
            gt_instances=sampled_gt_instances)
        pred_instances = InstanceData(scores=cls_score, masks=mask_pred)
        sampling_result = self.sampler.sample(
            assign_result=assign_result,
            pred_instances=pred_instances,
            gt_instances=gt_instances)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label target
        labels = gt_labels.new_full((self.instance_queries, ),0,dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        
        # if img_meta['use_inst']:
        #     label_weights = gt_labels.new_ones((self.instance_queries, ))
        # else:
        #     label_weights = gt_labels.new_zeros((self.instance_queries, ))
        label_weights = gt_labels.new_ones((self.instance_queries, ))

        # mask target
        mask_targets = gt_masks[sampling_result.pos_assigned_gt_inds]
        mask_weights = mask_pred.new_zeros((self.instance_queries, ))
        # if img_meta['use_inst']:
        #     mask_weights[pos_inds] = 1.0
        mask_weights[pos_inds] = 1.0

        return (labels, label_weights, mask_targets, mask_weights, pos_inds,
                neg_inds, sampling_result)

    def predict(self, dict_inputs: dict, databatch,mask_input=None, iou_threshold=0.7):
        """
        dict_inputs: dict, 
            vision_features: Tensor, (bs, c, h, w)
            vision_pos_enc: List[Tensor]: [bs, c, h1,w1]...
            backbone_fpn: List[Tensor]: [bs, c, h1,w1]...
        """
        pred_mask_logits, pred_cls_logits = self(dict_inputs, mask_input)
        bs = databatch['images'].shape[0]
        pred_bboxes = []

        for i in range(bs):
            # 1. 分类 logits 做 softmax
            cls_probs = F.softmax(pred_cls_logits[i], dim=-1)  # (num_queries, num_classes)
            scores, labels = cls_probs.max(dim=-1)  # (num_queries,), (num_queries,)

            # 2. 筛选 正类（label > 0）
            keep = labels > 0
            if keep.sum() == 0:
                pred_bboxes.append([])  # 当前图无预测
                continue
            
            scores = scores[keep]
            labels = labels[keep]
            mask_logits = pred_mask_logits[i][keep]  # (num_keep, h, w)

            # 3. mask logits resize到(H, W)，再 sigmoid + 二值化
            ori_size = databatch['metainfo'][i]['origin_size']
            mask_logits_resized = F.interpolate(mask_logits.unsqueeze(1), size=ori_size, mode='bilinear', align_corners=False).squeeze(1)  # (num_keep, H, W)
            masks = mask_logits_resized.sigmoid() > 0.5

            # 4. 过滤掉空 mask
            valid = masks.flatten(1).sum(dim=1) > 0  # (num_keep,), mask有前景的
            if valid.sum() == 0:
                pred_bboxes.append([])
                continue
            
            masks = masks[valid]
            scores = scores[valid]
            labels = labels[valid]

            # 5. 根据有效 masks 批量计算 bbox
            boxes = self.masks_to_boxes(masks)  # (num_valid, 4)

            # 6. NMS（不考虑类别）
            keep_idx = nms(boxes, scores, iou_threshold)

            boxes = boxes[keep_idx]
            scores = scores[keep_idx]
            labels = labels[keep_idx]
            masks = masks[keep_idx]

            image_boxes = []
            for box, score, label, mask in zip(boxes, scores, labels, masks):
                x1, y1, x2, y2 = box.tolist()
                image_boxes.append({
                    'bbox': [x1, y1, x2, y2],
                    # 'mask': mask.detach().cpu(),
                    'score': score.item(),
                    'cls': label.item()
                })
            
            pred_bboxes.append(image_boxes)

        return pred_bboxes
    
    def masks_to_boxes(self, masks):
        """
        批量根据二值 masks 计算 bounding boxes
        masks: (N, H, W), float tensor, 0 or 1
        返回: (N, 4), 每个是 [xmin, ymin, xmax, ymax]
        """
        N, H, W = masks.shape
        boxes = torch.zeros((N, 4), device=masks.device, dtype=torch.float)

        # 水平方向：每行是否有前景
        rows = masks.any(dim=2)  # (N, H)
        # 垂直方向：每列是否有前景
        cols = masks.any(dim=1)  # (N, W)

        for i in range(N):
            if rows[i].any():
                ymin, ymax = rows[i].nonzero(as_tuple=False)[[0, -1], 0]
                xmin, xmax = cols[i].nonzero(as_tuple=False)[[0, -1], 0]
                boxes[i] = torch.tensor([xmin, ymin, xmax, ymax], device=masks.device, dtype=torch.float)
            else:
                # 空mask，保持为0,0,0,0
                boxes[i] = torch.tensor([0, 0, 0, 0], device=masks.device, dtype=torch.float)

        return boxes
