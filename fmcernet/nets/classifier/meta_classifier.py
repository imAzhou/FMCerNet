import torch
import torch.nn as nn
from abc import abstractmethod

class MetaClassifier(nn.Module):
    def __init__(self, evaluator, args):
        super(MetaClassifier, self).__init__()
        self.evaluator = evaluator
        self.backbone_type = args.backbone_type

    @property
    def device(self):
        return next(self.parameters()).device

    def get_mlc_labels(self, databatch):
        bs = databatch['inputs'].shape[0]
        # binary_matrix: (bs, pos_cls)
        binary_matrix = torch.zeros((bs, self.num_classes), dtype=torch.float32).to(self.device)
        for idx, sample in enumerate(databatch['data_samples']):
            labels = sample.gt_label
            if len(labels) > 0:  # 非空才更新
                binary_matrix[idx, torch.as_tensor(labels, dtype=torch.long)] = 1.0
        return binary_matrix
    
    def get_img_tokens(self, inputs):
        '''Return img tokens: (bs, num_tokens, C)'''
        if self.backbone_type in ['resnet', 'convnext']:
            img_tokens = inputs[-1]   # (bs,c,h,w)
            img_tokens = img_tokens.flatten(2).transpose(1, 2)
        elif self.backbone_type == 'sam':
            img_tokens = inputs   # (bs,h*w,c)
        elif self.backbone_type == 'sam2':
            img_tokens = inputs['trunk_outputs'][-1]   # (bs,c,h,w)
            img_tokens = img_tokens.flatten(2).transpose(1, 2)
        elif self.backbone_type in ['vit','dinov2', 'dinov3', 'uni']:
            img_tokens = inputs[:,1:,:]
        elif self.backbone_type == 'uni2-h':
            img_tokens = inputs[:,9:,:]
        elif self.backbone_type == 'ctranspath':
            img_tokens = inputs
        elif self.backbone_type in ['smartccs', 'cytofm', 'unicas', 'virchow', 'virchow2', 'gpfm', 'genbio-pathfm']:
            img_tokens = inputs['x_norm_patchtokens']
        elif self.backbone_type == 'fusionnet':
            # feat_1 = inputs['x_norm_patchtokens']
            # feat_2 = inputs['dtcwt_output']
            # img_tokens = feat_1 + feat_2
            img_tokens = inputs['cat_output']
        return img_tokens
    
    def get_cls_token(self, inputs):
        '''Return cls token: (bs, C)'''
        if self.backbone_type in ['resnet', 'convnext']:
            feat = inputs[-1]
            cls_token = feat.mean(dim=[2, 3])
        elif self.backbone_type == 'sam2':
            feat = inputs['vision_features']
            cls_token = feat.mean(dim=[2, 3])
        elif self.backbone_type in ['vit', 'dinov2', 'dinov3', 'uni', 'uni2-h']:
            cls_token = inputs[:,0,:]
        elif self.backbone_type == 'ctranspath':
            cls_token = inputs.mean(dim=1)
        elif self.backbone_type in ['smartccs', 'cytofm', 'unicas', 'virchow', 'virchow2', 'gpfm', 'genbio-pathfm']:
            cls_token = inputs['x_norm_clstoken']
        elif self.backbone_type == 'fusionnet':
            vit_cls = inputs['x_norm_clstoken']
            dtcwt_mean = inputs['dtcwt_output'].mean(dim=1)  # (B, C)
            cls_token = vit_cls + dtcwt_mean
            # cls_token = torch.cat([vit_cls, dtcwt_mean], dim=1)  # (B, C1+C2)
        return cls_token

    @abstractmethod
    def calc_logits(self, x: torch.Tensor):
        '''
        Args:
            x (tensor): input tensor
        Return:
            logits result (tuple)
        '''
    
    @abstractmethod
    def calc_loss(self, x, databatch):
        '''
        Args:
            x (tensor): input tensor
            databatch (dict): input with GT info
        Return:
            loss (float): loss can be back propagation
        '''
    
    @abstractmethod
    def set_pred(self, x, databatch):
        '''
        Args:
            x (tensor): input tensor
            databatch (dict): input with GT info
        Return:
            databatch (dict): update with Predict info
        '''
    
    
