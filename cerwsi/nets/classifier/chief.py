import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from types import SimpleNamespace
from .meta_classifier import MetaClassifier
from cerwsi.utils import build_evaluator,BinaryMetric


class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x

class CHIEF(MetaClassifier):
    '''/medical-data/lhr/Prostate/Gleason/code/穿刺数据二次微调_二分类_2/models'''
    def __init__(self, args):
        num_classes = 1 # 只能做阴阳二分类
        evaluator = build_evaluator([BinaryMetric(args.logger_name, 
                                                  thr = args.positive_thr,
                                                  save_result_dir = args.save_result_dir,)])
        super(CHIEF, self).__init__(evaluator, args)
        
        self.backbone_type = args.backbone_type
        input_embed_dim = args.backbone_cfg['backbone_output_dim'][-1]
        if input_embed_dim == 384:
            size = [384, 256, 256]
        elif input_embed_dim == 768:
            size = [768, 512, 256]
        elif input_embed_dim == 1024:
            size = [1024, 512, 384]
        elif input_embed_dim == 2048:
            size = [2048, 1024, 512]
        else:
            size = [input_embed_dim, 512, 256]

        fc = [nn.Linear(size[0], size[1]), 
            nn.ReLU(),
            nn.Dropout(0.25)]
        attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=True, n_classes=num_classes)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], 1)


    def calc_logits(self, inputs):
        if self.backbone_type == 'resnet':
            feat = inputs[-1]   # (bs,c,h,w)
            feat = feat.flatten(2).transpose(1, 2)
        elif self.backbone_type == 'sam':
            feat = inputs   # (bs,h*w,c)
        elif self.backbone_type == 'sam2':
            feat = inputs['vision_features']   # (bs,c,h,w)
            feat = feat.flatten(2).transpose(1, 2)
        elif self.backbone_type in ['dinov2', 'uni']:
            feat = inputs[:,1:,:]
        elif self.backbone_type == 'smartccs':
            feat = inputs['x_norm_patchtokens']
        
        # feat: (bs,img_token,C)
        # A: (bs, num_tokens, pos_cls_num), h: (bs, num_tokens, c=512)
        A, h = self.attention_net(feat)
        A_raw = A    # A_raw: (bs, num_tokens, 1)
        A = A.transpose(1, 2)    # A: (bs, 1, num_tokens)
        A = F.softmax(A, dim=-1)
        cls_feature = torch.bmm(A, h)    # cls_feature: (bs, 1, c=512)
        out = self.classifiers(cls_feature)    # (bs, 1, 1)
        # img_embedding = torch.mm(A, feat).squeeze(1)
        inter_var = {
            'A_raw': A_raw,
            'token_feats': h,
            'img_feat': cls_feature.squeeze(1),  # cls_feature: (bs, c=512)
            # 'img_feat': img_embedding
        }
        return out.squeeze(-1), inter_var
    
    def calc_loss(self,inputs, databatch):
        img_pn_logit,_ = self.calc_logits(inputs)
        image_labels = torch.tensor([int(len(item.gt_label)>0) for item in databatch['data_samples']])
        img_gt = image_labels.to(self.device).unsqueeze(-1).float()
        pn_loss = F.binary_cross_entropy_with_logits(img_pn_logit, img_gt, reduction='mean')
        loss_dict = {
            'pn_loss': pn_loss.item(),
        }
        return pn_loss,loss_dict

    def set_pred(self,inputs, databatch):
        img_pn_logit,inter_var = self.calc_logits(inputs) # (bs, num_classes-1)
        img_probs = torch.sigmoid(img_pn_logit).squeeze(-1)   # (bs, )
        bs = len(databatch['data_samples'])
        if bs == 1:
            img_probs = img_probs.unsqueeze(0)

        data_sampels = []
        patch_prob = (self.patch_probs(inter_var))['patch_prob']
        for item, pn_p,attn,imgtoken in zip(databatch['data_samples'], img_probs, patch_prob, inter_var['img_feat']):
            item.img_prob = pn_p
            item.attn = attn    # (num_tokens,)
            item.img_token = imgtoken
            data_sampels.append(item)
        return data_sampels

    def patch_probs(self, inter_var):
        A_raw = inter_var['A_raw']  # A_raw: (bs, num_tokens, 1)
        h = inter_var['token_feats']    # h: (bs, num_tokens, c=512)
        patch_logits = torch.sigmoid(self.classifiers(h))    # h: (bs, num_tokens, 1)
        patch_prob = torch.sigmoid(A_raw.squeeze()) * patch_logits.squeeze()

        return {
            'patch_prob': patch_prob,  # A_raw: (bs, num_tokens)
            'attention_raw': A_raw.squeeze()  # A_raw: (bs, num_tokens)
        }
    