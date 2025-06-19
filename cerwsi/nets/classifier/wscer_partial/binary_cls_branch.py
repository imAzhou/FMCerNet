import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [nn.Linear(L, D), nn.Tanh()]
        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
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

class BinaryClsBranch(nn.Module):
    def __init__(self, input_dim):
        num_classes = 1 # 只能做阴阳二分类
        super(BinaryClsBranch, self).__init__()
        size = [input_dim, 512, 256]

        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(0.25)]
        attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=True, n_classes=num_classes)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], 1)


    def forward(self, vision_features: torch.Tensor):
        '''
        Args:
            vision_features: (bs, c, h, w)
        Return:
            img_logits: (bs, 1)
        '''
        # img_tokens: (bs,img_token,C)
        img_tokens = vision_features.flatten(2).transpose(1,2)
        # A: (bs, num_tokens, pos_cls_num), h: (bs, num_tokens, c=512)
        A, h = self.attention_net(img_tokens)
        A_raw = A    # A_raw: (bs, num_tokens, 1)
        A = A.transpose(1, 2)    # A: (bs, 1, num_tokens)
        A = F.softmax(A, dim=-1)
        cls_feature = torch.bmm(A, h)    # cls_feature: (bs, 1, c=512)
        out = self.classifiers(cls_feature)    # (bs, 1, 1)
        inter_var = {
            'A_raw': A_raw,
            'token_feats': h,
        }
        return out.squeeze(-1), inter_var

    def patch_probs(self, inter_var, scale):
        A_raw = inter_var['A_raw']  # A_raw: (bs, num_tokens, 1)
        h = inter_var['token_feats']    # h: (bs, num_tokens, c=512)
        with torch.no_grad():
            patch_logits = torch.sigmoid(self.classifiers(h))    # h: (bs, num_tokens, 1)
            patch_probs = torch.sigmoid(A_raw.squeeze()) * patch_logits.squeeze()

        bs,num_tokens = patch_probs.shape
        feat_size = int(math.sqrt(num_tokens))
        patch_probs = patch_probs.reshape((bs, feat_size, feat_size)).unsqueeze(1).detach()
        patch_probs = F.interpolate(patch_probs, size=(feat_size*scale, feat_size*scale), mode='bilinear')

        # return {
        #     'patch_prob': patch_prob,  # A_raw: (bs, num_tokens)
        #     'attention_raw': A_raw.squeeze()  # A_raw: (bs, num_tokens)
        # }
        return patch_probs   # (bs, 1, h, w)