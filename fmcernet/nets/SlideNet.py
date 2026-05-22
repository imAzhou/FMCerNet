import torch
import torch.nn as nn
from fmcernet.utils import build_evaluator, SlideMetric
from mmengine.optim import OptimWrapper
from .get_mil import get_mil

class SlideNet(nn.Module):
    def __init__(self, cfg):
        super(SlideNet, self).__init__()
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
    
    def train_step(self, databatch, optim_wrapper: OptimWrapper):
        loss,loss_dict = self.mil_model.calc_loss(databatch)
        optim_wrapper.update_params(loss)
        return loss,loss_dict

    def val_step(self, databatch):
        loss, loss_dict = self.mil_model.calc_loss(databatch)
        databatch = self.mil_model.set_pred(databatch)
        return databatch, loss, loss_dict
