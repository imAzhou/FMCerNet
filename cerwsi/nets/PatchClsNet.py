import torch
import torch.nn as nn

from mmengine.optim import OptimWrapper
from .get_backbone import get_backbone
from .get_neck import get_neck
from .get_classifier import get_classifier

class PatchClsNet(nn.Module):
    def __init__(self, cfg):
        super(PatchClsNet, self).__init__()

        self.backbone = get_backbone(cfg)
        self.neck_type = cfg.neck_type
        if self.neck_type is not None:
            self.neck = get_neck(cfg)
        self.classifier = get_classifier(cfg)

        frozen_backbone = cfg.backbone_cfg['frozen_backbone']
        use_peft = cfg.backbone_cfg['use_peft']
        self.backbone_nograd = frozen_backbone and use_peft is None
        pixel_mean = [123.675, 116.28, 103.53]
        pixel_std = [58.395, 57.12, 57.375]
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        
    @property
    def device(self):
        return next(self.parameters()).device

    def load_ckpt(self, ckpt):
        params_weight = torch.load(ckpt, map_location=self.device)
        print(self.load_state_dict(params_weight, strict=True))
    
    def forward(self, data_batch, mode, optim_wrapper=None):        
        if mode == 'train':
            return self.train_step(data_batch, optim_wrapper)
        if mode == 'val':
            return self.val_step(data_batch)
    
    def extract_feature(self, input_x):
        # input_x = input_x[:, [2, 1, 0], :, :]   # bgr2rgb
        # input_x = (input_x - self.pixel_mean) / self.pixel_std  # color norm
        feature_emb = self.backbone(input_x)
        return feature_emb

    def train_step(self, databatch, optim_wrapper: OptimWrapper):
        input_x = databatch['inputs']   # (bs, c, h, w)
        feature_emb = self.extract_feature(input_x)
        # feature_emb = self.backbone(input_x)   # (bs, c, h, w) or (bs, c, numtokens) or dict
        if self.neck_type is not None:
            feature_emb = self.neck(feature_emb)
        loss,loss_dict = self.classifier.calc_loss(feature_emb, databatch)
        optim_wrapper.update_params(loss)
        return loss,loss_dict

    def val_step(self, databatch):
        input_x = databatch['inputs']
        feature_emb = self.extract_feature(input_x)
        # feature_emb = self.backbone(input_x)
        if self.neck_type is not None:
            feature_emb = self.neck(feature_emb)
        databatch = self.classifier.set_pred(feature_emb, databatch)
        return databatch
