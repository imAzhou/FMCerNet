import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple
from mmengine.structures import InstanceData
from mmdet.models.task_modules import MaxIoUAssigner,RandomSampler
from mmengine.optim import OptimWrapper
import numpy as np
from cerwsi.utils import build_evaluator,ImgODMetric,ImgODCOCOMetric
from torchvision.ops import nms
from sam2.build_sam import build_sam2
from sam2.utils.amg import build_all_layer_point_grids

class InferSegNet(nn.Module):
    def __init__(self, cfg):
        super(InferSegNet, self).__init__()
        self.points_per_batch = 64
        self.point_grids = build_all_layer_point_grids(
            n_per_side = 64,
            n_layers = 0,
            scale_per_layer = 1,
        )
        backbone_ckpt = cfg.backbone_cfg['backbone_ckpt']
        # backbone_config_file = cfg.backbone_cfg['config_file']
        backbone_config_file = "configs/sam2.1/sam2.1_hiera_l.yaml"

        self.model = build_sam2(backbone_config_file, backbone_ckpt, apply_postprocessing=False)
        pixel_mean = [123.675, 116.28, 103.53]
        pixel_std = [58.395, 57.12, 57.375]
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        self._bb_feat_sizes = [(256, 256),(128, 128),(64, 64)]
        self.bbox_pos_thr = 0.5
        input_embed_dim = cfg.backbone_cfg['backbone_output_dim'][-1]
        self.classifier = nn.Linear(input_embed_dim, 1)
        save_result_dir = getattr(cfg, 'save_result_dir', None)
        evaluator = build_evaluator([ImgODCOCOMetric(
            cfg.logger_name,save_result_dir,
            cfg.val_evaluator,cfg.classes
        )])
        self.classifier.evaluator = evaluator

        self.assigner = MaxIoUAssigner(
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
        )

        self.freeze_sam2()
        
    @property
    def device(self):
        return next(self.parameters()).device

    def load_ckpt(self, ckpt):
        params_weight = torch.load(ckpt, map_location=self.device)
        print(self.classifier.load_state_dict(params_weight, strict=True))
    
    def freeze_sam2(self):
        for name, param in self.model.named_parameters():
            param.requires_grad = False


    def forward(self, data_batch, mode, optim_wrapper=None):        
        if mode == 'train':
            return self.train_step(data_batch, optim_wrapper)
        if mode == 'val':
            return self.val_step(data_batch)
    
    def extract_feature(self, input_x):
        input_x = input_x[:, [2, 1, 0], :, :]   # bgr2rgb
        input_x = (input_x - self.pixel_mean) / self.pixel_std  # color norm
        backbone_out = self.model.forward_image(input_x)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
        # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed
        batch_size = input_x.shape[0]
        feats = [
            feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]
        self._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}

    def infer_proposal(self, databatch):
        topk = 100
        proposals_list = []
        for datasample in databatch['data_samples']:
            sam2proposal = datasample.get('sam2proposal', None)
            if sam2proposal:
                N = sam2proposal['scores'].shape[0]
                sorted_indices = torch.argsort(sam2proposal['scores'], descending=True)
                sample_indices = sorted_indices[torch.randint(0, N, (topk,))] if N < topk else sorted_indices[:topk]
                top_bboxes = sam2proposal['bboxes'][sample_indices]
                # top_scores = score[sample_indices]
                proposals_list.append(top_bboxes)
            # TODO: everything mode to generate proposal
                
        return proposals_list
    
    def match_proposal_gt(self, databatch, proposal_bboxes):
        assign_labellist = []
        for bboxes, datasample in zip(proposal_bboxes, databatch['data_samples']):
            pred_instances = InstanceData()
            pred_instances.priors = bboxes
            assign_result = self.assigner.assign(pred_instances, datasample.gt_instances)
            label_list = (assign_result.labels>0).int()
            assign_labellist.append(label_list)
        
        return torch.stack(assign_labellist).to(self.device)
    
    def calc_proposal_feat(self, input_len, proposals):
        bs, c, h, w = self._features['image_embed'].shape
        feat_len = h
        scale_factor = input_len // feat_len
        scale_proposals = torch.stack(proposals) / scale_factor     # (bs,num_bboxes,4), 4 means (x1,y1,x2,y2)
        xy1 = torch.floor(scale_proposals[..., :2])      # (bs, num_bboxes, 2)
        xy2 = torch.ceil(scale_proposals[..., 2:])     # (bs, num_bboxes, 2)
        bboxes_int = torch.cat([xy1, xy2], dim=-1).to(torch.int)
        feat_vector = []
        for feat,bboxes in zip(self._features['image_embed'], bboxes_int):
            per_image_vecs = []
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox.tolist()
                # 安全裁剪，防止越界
                x1 = max(0, min(x1, w - 1))
                x2 = max(0, min(x2, w))
                y1 = max(0, min(y1, h - 1))
                y2 = max(0, min(y2, h))

                # 裁剪区域
                region = feat[:, y1:y2, x1:x2]  # (c, h', w')
                if region.numel() == 0:
                    vec = torch.zeros(c)
                else:
                    vec = region.mean(dim=[1, 2])  # 平均池化成 (c,)

                per_image_vecs.append(vec)
            
            feat_vector.append(torch.stack(per_image_vecs))  # (num_bboxes, c)

        # 最终输出： (bs, num_bboxes, c)
        output = torch.stack(feat_vector)
        return output

    def train_step(self, databatch, optim_wrapper: OptimWrapper):
        input_x = databatch['inputs']   # (bs, c, h, w)
        self.extract_feature(input_x)
        proposals = self.infer_proposal(databatch)
        proposal_gts = self.match_proposal_gt(databatch, proposals)  # (bs, k)   # 0/1: is lesion or not
        input_len = databatch['inputs'].shape[-1]
        proposal_feats = self.calc_proposal_feat(input_len, proposals)  # (bs, k, C)
        pred_logits = self.classifier(proposal_feats)  # (bs, k, 1)
        proposal_loss = F.binary_cross_entropy_with_logits(
            pred_logits.squeeze(-1), proposal_gts.float(), reduction='mean')
        loss_dict = {
            'proposal_loss': proposal_loss.item(),
        }
        optim_wrapper.update_params(proposal_loss)
        return proposal_loss,loss_dict

    def val_step(self, databatch):
        input_x = databatch['inputs']
        self.extract_feature(input_x)
        proposals = self.infer_proposal(databatch)  # size in input scale
        input_len = databatch['inputs'].shape[-1]
        proposal_feats = self.calc_proposal_feat(input_len, proposals)  # (bs, k, C)
        pred_scores = torch.sigmoid(self.classifier(proposal_feats))  # (bs, k, 1)
        
        pred_bboxes,img_probs = [],[]
        for proposal_bboxes,proposal_scores,datasample in zip(proposals, pred_scores, databatch['data_samples']):
            bboxes,scores = [],[]
            for box, score in zip(proposal_bboxes, proposal_scores):
                if score > self.bbox_pos_thr:
                    sf = datasample.scale_factor
                    x1, y1, x2, y2 = box.tolist()
                    x1, y1, x2, y2 = int(x1/sf[0]), int(y1/sf[1]), int(x2/sf[0]), int(y2/sf[1])
                    bboxes.append([x1, y1, x2, y2])
                    scores.append(score.item())
            iou_threshold = 0.5
            bboxes, scores = torch.Tensor(bboxes),torch.Tensor(scores)
            keep_idx = nms(bboxes, scores, iou_threshold)
            bboxes = bboxes[keep_idx]
            scores = scores[keep_idx]

            image_boxes = [{'bbox':bbox, 'score':score, 'cls':1} for bbox, score in zip(bboxes, scores)]
            pred_bboxes.append(image_boxes)
            img_probs.append(1. if len(image_boxes)>0 else 0.)
        
        databatch['pred_bbox'] = pred_bboxes
        databatch['img_probs'] = torch.Tensor(img_probs)
        return databatch
    