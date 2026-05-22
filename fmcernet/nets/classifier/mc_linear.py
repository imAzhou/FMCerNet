import torch
from torch import nn
from .meta_classifier import MetaClassifier
from fmcernet.utils import build_evaluator, MultiClsMetric


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = (drop, drop)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class MCLinear(MetaClassifier):
    def __init__(self, args):
        evaluator = build_evaluator([MultiClsMetric(
            num_classes = args.num_classes,
            classes = args.classes,
            logger_name = args.logger_name,
        )])
        super(MCLinear, self).__init__(evaluator, args)
        input_embed_dim = args.backbone_cfg['backbone_output_dim'][-1]
        self.num_classes = args.num_classes
        self.backbone_type = args.backbone_type
        # self.cls_linear_heads = nn.Linear(input_embed_dim, self.num_classes)
        self.cls_linear_heads = Mlp(
            in_features = input_embed_dim, 
            hidden_features=input_embed_dim//2, 
            out_features=self.num_classes, 
            act_layer=nn.GELU, 
            drop=0.02
        )
    
    def calc_logits(self, inputs):
        cls_token = self.get_cls_token(inputs)
        pred_logits = self.cls_linear_heads(cls_token)  # （bs, num_cls）
        return pred_logits
    
    def calc_loss(self, inputs, databatch):
        loss_fn = nn.CrossEntropyLoss()
        pred_logits = self.calc_logits(inputs)
        gt_labels = torch.cat([item.gt_label for item in databatch['data_samples']]).to(self.device)
        loss = loss_fn(pred_logits, gt_labels)
        loss_dict = {
            'loss': loss.item(),
        }
        return loss,loss_dict

    def set_pred(self,inputs, databatch):
        pred_logits = self.calc_logits(inputs) # (bs, num_classes)
        pred_probs = torch.softmax(pred_logits, dim=-1) # (bs, n_cls)
        _,pred_labels = torch.max(pred_probs, dim=-1)
        data_sampels = []
        for item, pred_prob, pred_label in zip(databatch['data_samples'], pred_probs, pred_labels):
            item.pred_prob = pred_prob
            item.pred_label = pred_label
            data_sampels.append(item)

        return data_sampels
