from mmdet.models.layers import (DetrTransformerDecoder, DetrTransformerEncoder,
                      SinePositionalEncoding)
from mmdet.registry import MODELS
import torch.nn as nn
import torch
from collections import OrderedDict
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union
from mmengine.utils import is_list_of
from mmdet.utils import InstanceList
from mmdet.models.utils import samplelist_boxtype2tensor
from torch import Tensor
import math
from cerwsi.utils import WSCocoMetric
from einops import rearrange
from mmdet.structures import OptSampleList,SampleList

class WSDETR(nn.Module):
    def __init__(self, args):
        super(WSDETR, self).__init__()
        self.evaluator = WSCocoMetric(**args.val_evaluator)
        self.evaluator.dataset_meta = dict(classes=args.classes)
        # process args
        num_queries = 100
        self.backbone_type = args.backbone_type
        in_channel = args.backbone_cfg['backbone_output_dim'][-1]
        train_cfg=dict(
            assigner=dict(
                type='HungarianAssigner',
                match_costs=[
                    dict(type='ClassificationCost', weight=1.),
                    dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                    dict(type='IoUCost', iou_mode='giou', weight=2.0)
                ]))
        test_cfg=dict(max_per_img=100)
        neck=dict(
            type='ChannelMapper',
            in_channels=[in_channel],
            kernel_size=1,
            out_channels=256,
            act_cfg=None,
            norm_cfg=None,
            num_outs=1)
        bbox_head = dict(
            type='DETRHead',
            num_classes=args.num_classes,
            embed_dims=256,
            loss_cls=dict(
                type='CrossEntropyLoss',
                bg_cls_weight=0.1,
                use_sigmoid=False,
                loss_weight=1.0,
                class_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=5.0),
            loss_iou=dict(type='GIoULoss', loss_weight=2.0))
        encoder=dict(  # DetrTransformerEncoder
            num_layers=6,
            layer_cfg=dict(  # DetrTransformerEncoderLayer
                self_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=256,
                    num_heads=8,
                    dropout=0.1,
                    batch_first=True),
                ffn_cfg=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    ffn_drop=0.1,
                    act_cfg=dict(type='ReLU', inplace=True))))
        decoder=dict(  # DetrTransformerDecoder
            num_layers=6,
            layer_cfg=dict(  # DetrTransformerDecoderLayer
                self_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=256,
                    num_heads=8,
                    dropout=0.1,
                    batch_first=True),
                cross_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=256,
                    num_heads=8,
                    dropout=0.1,
                    batch_first=True),
                ffn_cfg=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    ffn_drop=0.1,
                    act_cfg=dict(type='ReLU', inplace=True))),
            return_intermediate=True)
        positional_encoding=dict(num_feats=128, normalize=True)

        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.encoder = encoder
        self.decoder = decoder
        self.positional_encoding = positional_encoding
        self.num_queries = num_queries

        # init model layers
        self.neck = MODELS.build(neck)
        self.bbox_head = MODELS.build(bbox_head)
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding)
        self.encoder = DetrTransformerEncoder(**self.encoder)
        self.decoder = DetrTransformerDecoder(**self.decoder)
        self.embed_dims = self.encoder.embed_dims
        # NOTE The embed_dims is typically passed from the inside out.
        # For example in DETR, The embed_dims is passed as
        # self_attn -> the first encoder layer -> encoder -> detector.
        self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)

        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'

    @property
    def device(self):
        return next(self.parameters()).device
    
    def forward_transformer(self,
                            img_feats: Tuple[Tensor],
                            batch_data_samples: OptSampleList = None) -> Dict:
        """Forward process of Transformer, which includes four steps:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'. We
        summarized the parameters flow of the existing DETR-like detector,
        which can be illustrated as follow:

        .. code:: text

                 img_feats & batch_data_samples
                               |
                               V
                      +-----------------+
                      | pre_transformer |
                      +-----------------+
                          |          |
                          |          V
                          |    +-----------------+
                          |    | forward_encoder |
                          |    +-----------------+
                          |             |
                          |             V
                          |     +---------------+
                          |     |  pre_decoder  |
                          |     +---------------+
                          |         |       |
                          V         V       |
                      +-----------------+   |
                      | forward_decoder |   |
                      +-----------------+   |
                                |           |
                                V           V
                               head_inputs_dict

        Args:
            img_feats (tuple[Tensor]): Tuple of feature maps from neck. Each
                    feature map has shape (bs, dim, H, W).
            batch_data_samples (list[:obj:`DetDataSample`], optional): The
                batch data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            dict: The dictionary of bbox_head function inputs, which always
            includes the `hidden_states` of the decoder output and may contain
            `references` including the initial and intermediate references.
        """
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples)

        encoder_outputs_dict = self.forward_encoder(**encoder_inputs_dict)

        tmp_dec_in, head_inputs_dict = self.pre_decoder(**encoder_outputs_dict)
        decoder_inputs_dict.update(tmp_dec_in)

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)
        return head_inputs_dict

    def process_backbone_outputs(self, outputs):
        if self.backbone_type == 'resnet':
            feat = outputs[-1]   # (bs,c,h,w)
        elif self.backbone_type == 'sam':
            fs = int(math.sqrt(outputs.shape[1]))
            feat = rearrange(outputs, 'b (h w) c -> b c h w', h=fs, w=fs)
        elif self.backbone_type == 'sam2':
            feat = outputs['trunk_outputs'][-1]   # (bs,c,h,w)
        elif self.backbone_type in ['dinov2', 'uni']:
            feat = outputs[:,1:,:]
            fs = int(math.sqrt(feat.shape[1]))
            feat = rearrange(feat, 'b (h w) c -> b c h w', h=fs, w=fs)
        elif self.backbone_type == 'smartccs':
            feat = outputs['x_norm_patchtokens']
            fs = int(math.sqrt(feat.shape[1]))
            feat = rearrange(feat, 'b (h w) c -> b c h w', h=fs, w=fs)
        return (feat,)

    def pre_transformer(
            self,
            img_feats: Tuple[Tensor],
            batch_data_samples: OptSampleList = None) -> Tuple[Dict, Dict]:
        """Prepare the inputs of the Transformer.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            img_feats (Tuple[Tensor]): Tuple of features output from the neck,
                has shape (bs, c, h, w).
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such as
                `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            tuple[dict, dict]: The first dict contains the inputs of encoder
            and the second dict contains the inputs of decoder.

            - encoder_inputs_dict (dict): The keyword args dictionary of
              `self.forward_encoder()`, which includes 'feat', 'feat_mask',
              and 'feat_pos'.
            - decoder_inputs_dict (dict): The keyword args dictionary of
              `self.forward_decoder()`, which includes 'memory_mask',
              and 'memory_pos'.
        """

        feat = img_feats[-1]  # NOTE img_feats contains only one feature.
        batch_size, feat_dim, _, _ = feat.shape
        # construct binary masks which for the transformer.
        assert batch_data_samples is not None
        batch_input_shape = batch_data_samples[0].batch_input_shape
        input_img_h, input_img_w = batch_input_shape
        img_shape_list = [sample.img_shape for sample in batch_data_samples]
        same_shape_flag = all([
            s[0] == input_img_h and s[1] == input_img_w for s in img_shape_list
        ])
        if torch.onnx.is_in_onnx_export() or same_shape_flag:
            masks = None
            # [batch_size, embed_dim, h, w]
            pos_embed = self.positional_encoding(masks, input=feat)
        else:
            masks = feat.new_ones((batch_size, input_img_h, input_img_w))
            for img_id in range(batch_size):
                img_h, img_w = img_shape_list[img_id]
                masks[img_id, :img_h, :img_w] = 0
            # NOTE following the official DETR repo, non-zero values represent
            # ignored positions, while zero values mean valid positions.

            masks = F.interpolate(
                masks.unsqueeze(1),
                size=feat.shape[-2:]).to(torch.bool).squeeze(1)
            # [batch_size, embed_dim, h, w]
            pos_embed = self.positional_encoding(masks)

        # use `view` instead of `flatten` for dynamically exporting to ONNX
        # [bs, c, h, w] -> [bs, h*w, c]
        feat = feat.view(batch_size, feat_dim, -1).permute(0, 2, 1)
        pos_embed = pos_embed.view(batch_size, feat_dim, -1).permute(0, 2, 1)
        # [bs, h, w] -> [bs, h*w]
        if masks is not None:
            masks = masks.view(batch_size, -1)

        # prepare transformer_inputs_dict
        encoder_inputs_dict = dict(
            feat=feat, feat_mask=masks, feat_pos=pos_embed)
        decoder_inputs_dict = dict(memory_mask=masks, memory_pos=pos_embed)
        return encoder_inputs_dict, decoder_inputs_dict

    def forward_encoder(self, feat: Tensor, feat_mask: Tensor,
                        feat_pos: Tensor) -> Dict:
        """Forward with Transformer encoder.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            feat (Tensor): Sequential features, has shape (bs, num_feat_points,
                dim).
            feat_mask (Tensor): ByteTensor, the padding mask of the features,
                has shape (bs, num_feat_points).
            feat_pos (Tensor): The positional embeddings of the features, has
                shape (bs, num_feat_points, dim).

        Returns:
            dict: The dictionary of encoder outputs, which includes the
            `memory` of the encoder output.
        """
        memory = self.encoder(
            query=feat, query_pos=feat_pos,
            key_padding_mask=feat_mask)  # for self_attn
        encoder_outputs_dict = dict(memory=memory)
        return encoder_outputs_dict

    def pre_decoder(self, memory: Tensor) -> Tuple[Dict, Dict]:
        """Prepare intermediate variables before entering Transformer decoder,
        such as `query`, `query_pos`.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).

        Returns:
            tuple[dict, dict]: The first dict contains the inputs of decoder
            and the second dict contains the inputs of the bbox_head function.

            - decoder_inputs_dict (dict): The keyword args dictionary of
              `self.forward_decoder()`, which includes 'query', 'query_pos',
              'memory'.
            - head_inputs_dict (dict): The keyword args dictionary of the
              bbox_head functions, which is usually empty, or includes
              `enc_outputs_class` and `enc_outputs_class` when the detector
              support 'two stage' or 'query selection' strategies.
        """

        batch_size = memory.size(0)  # (bs, num_feat_points, dim)
        query_pos = self.query_embedding.weight
        # (num_queries, dim) -> (bs, num_queries, dim)
        query_pos = query_pos.unsqueeze(0).repeat(batch_size, 1, 1)
        query = torch.zeros_like(query_pos)

        decoder_inputs_dict = dict(
            query_pos=query_pos, query=query, memory=memory)
        head_inputs_dict = dict()
        return decoder_inputs_dict, head_inputs_dict

    def forward_decoder(self, query: Tensor, query_pos: Tensor, memory: Tensor,
                        memory_mask: Tensor, memory_pos: Tensor) -> Dict:
        """Forward with Transformer decoder.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            query (Tensor): The queries of decoder inputs, has shape
                (bs, num_queries, dim).
            query_pos (Tensor): The positional queries of decoder inputs,
                has shape (bs, num_queries, dim).
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).
            memory_mask (Tensor): ByteTensor, the padding mask of the memory,
                has shape (bs, num_feat_points).
            memory_pos (Tensor): The positional embeddings of memory, has
                shape (bs, num_feat_points, dim).

        Returns:
            dict: The dictionary of decoder outputs, which includes the
            `hidden_states` of the decoder output.

            - hidden_states (Tensor): Has shape
              (num_decoder_layers, bs, num_queries, dim)
        """

        hidden_states = self.decoder(
            query=query,
            key=memory,
            value=memory,
            query_pos=query_pos,
            key_pos=memory_pos,
            key_padding_mask=memory_mask)  # for cross_attn

        head_inputs_dict = dict(hidden_states=hidden_states)
        return head_inputs_dict

    def calc_loss(self, backbone_outputs: dict, databatch):
        feats = self.process_backbone_outputs(backbone_outputs)
        img_feats = self.neck(feats)
        data_samples = [item.to(self.device) for item in databatch['data_samples']]
        samplelist_boxtype2tensor(data_samples)
        head_inputs_dict = self.forward_transformer(img_feats,data_samples)
        loss_dict = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=data_samples)
        
        loss, log_vars = self.parse_losses(loss_dict)
        return loss, log_vars
    
    def parse_losses(
        self, losses: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Parses the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: There are two elements. The first is the
            loss tensor passed to optim_wrapper which may be a weighted sum
            of all losses, and the second is log_vars which will be sent to
            the logger.
        """
        log_vars = []
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars.append([loss_name, loss_value.mean()])
            elif is_list_of(loss_value, torch.Tensor):
                log_vars.append(
                    [loss_name,
                     sum(_loss.mean() for _loss in loss_value)])
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(value for key, value in log_vars if 'loss' in key)
        log_vars.insert(0, ['loss', loss])
        log_vars = OrderedDict(log_vars)  # type: ignore

        return loss, log_vars  # type: ignore

    def set_pred(self, backbone_outputs, databatch):
        """

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances'. And the ``pred_instances`` usually
            contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        img_feats = self.neck(self.process_backbone_outputs(backbone_outputs))
        data_samples = [item.to(self.device) for item in databatch['data_samples']]
        samplelist_boxtype2tensor(data_samples)
        head_inputs_dict = self.forward_transformer(img_feats,data_samples)
        results_list = self.bbox_head.predict(
            **head_inputs_dict,
            rescale=True,
            batch_data_samples=data_samples)
        for data_sample, pred_instances in zip(data_samples, results_list):
            data_sample.pred_instances = pred_instances
        samplelist_boxtype2tensor(data_samples)
        
        return [i.to_dict() for i in data_samples]
    