import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention


class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        return x

class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x

class TransMIL(nn.Module):
    def __init__(self, args):
        super(TransMIL, self).__init__()
        n_classes = args.num_classes
        C_in = args.C_in
        C_hidden = args.C_hidden
        self.pos_layer = PPEG(dim=C_hidden)
        self._fc1 = nn.Sequential(nn.Linear(C_in, C_hidden), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, C_hidden))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=C_hidden)
        self.layer2 = TransLayer(dim=C_hidden)
        self.norm = nn.LayerNorm(C_hidden)
        self._fc2 = nn.Linear(C_hidden, self.n_classes)

    @property
    def device(self):
        return next(self.parameters()).device
    
    def forward(self, input_x: torch.Tensor):
        '''
        Args:
            input_x: Tensor, [B, n, C_in]
        '''
        h = input_x.float() # [B, n, C_in]
        h = self._fc1(h) # [B, n, C_hidden]
        #---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]],dim = 1) # [B, N, 512]
        #---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)
        #---->Translayer x1
        h = self.layer1(h) #[B, N, 512]
        #---->PPEG
        h = self.pos_layer(h, _H, _W) #[B, N, 512]
        #---->Translayer x2
        h = self.layer2(h) #[B, N, 512]
        #---->cls_token
        h = self.norm(h)[:,0]
        #---->predict
        logits = self._fc2(h) #[B, n_classes]
        # Y_hat = torch.argmax(logits, dim=1)
        # Y_prob = F.softmax(logits, dim = 1)
        # results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        return logits
    
    def calc_loss(self, databatch):
        input_x = databatch['inputs']   # (bs, k, c)
        pred_logits = self.forward(input_x)
        loss_fn = nn.CrossEntropyLoss()
        label = torch.as_tensor([item['slide_label'] for item in databatch['data_samples']]).to(self.device)
        loss = loss_fn(pred_logits, label)
        return loss, {'bce_loss': loss}
    
    def set_pred(self, databatch):
        input_x = databatch['inputs']   # (bs, k, c)
        pred_logits = self.forward(input_x)
        pred_labels = torch.argmax(pred_logits, dim=1)
        pred_probs = F.softmax(pred_logits, dim = 1)
        data_sampels = []
        for item, label, prob in zip(databatch['data_samples'], pred_labels, pred_probs):
            item['pred_label'] = label
            item['pred_prob'] = prob
            data_sampels.append(item)

        return data_sampels

if __name__ == "__main__":
    data = torch.randn((1, 6000, 1024)).cuda()
    model = TransMIL(n_classes=2).cuda()
    print(model.eval())
    results_dict = model(data = data)
    print(results_dict)
