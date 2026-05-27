import torch
import torch.nn as nn
import torch.nn.functional as F

class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [nn.Linear(L, D),nn.Tanh()]

        self.attention_b = [nn.Linear(L, D),nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)  # N x D
        A = self.attention_c(A)  # N x n_classes
        return A, x

class CHIEF(nn.Module):
    def __init__(self, input_embed_dim):
        num_classes = 1 # 只能做阴阳二分类
        super(CHIEF, self).__init__()
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
            nn.Dropout(0.25)
        ]
        attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=True, n_classes=num_classes)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], 1)


    def forward(self, feat):
        '''feat: (bs,img_token,C)'''
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
    

    def patch_probs(self, inter_var):
        A_raw = inter_var['A_raw']  # A_raw: (bs, num_tokens, 1)
        h = inter_var['token_feats']    # h: (bs, num_tokens, c=512)
        patch_logits = torch.sigmoid(self.classifiers(h))    # h: (bs, num_tokens, 1)
        patch_prob = torch.sigmoid(A_raw.squeeze(-1)) * patch_logits.squeeze(-1)

        return {
            'patch_prob': patch_prob,  # A_raw: (bs, num_tokens)
            'attention_raw': A_raw.squeeze(-1)  # A_raw: (bs, num_tokens)
        }
    
