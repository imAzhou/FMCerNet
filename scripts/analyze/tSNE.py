import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import numpy as np
import matplotlib.colors as mcolors

def format_slide_tensor(feat_path, patch_nums, C_in):
    if os.path.exists(feat_path):
        load_tensor = torch.load(feat_path)    # (L, dim)
        pn_prob_feat = load_tensor[:,0,:]
        pn_prob, pn_feat = pn_prob_feat[:,0], pn_prob_feat[:,1:]
        pos_prob_feat = load_tensor[:,1:,:]
        pos_prob, pos_feat = pos_prob_feat[:,:,0], pos_prob_feat[:,:,1:]
        # step1: 按pn_prob从大到小排序
        sorted_idx = torch.argsort(pn_prob, descending=True)  # (L,)
        pn_feat_sorted = pn_feat[sorted_idx]                  # (L, dim)

        # step2: 取pos_prob前k个最大值对应的pos_feat
        top_idx = torch.topk(pos_prob, k=1, dim=1).indices    # (L, 3)
        top_pos_feat = torch.gather(pos_feat, 1, top_idx.unsqueeze(-1).expand(-1, -1, pos_feat.size(-1)))  # (L, 3, dim)
        pos_feat_sum = top_pos_feat.sum(dim=1)   # (L, dim)

        feat_concat = torch.cat([pn_feat_sorted, pos_feat_sum], dim=1)   # (L, dim*2)
        slide_tensor = feat_concat[sorted_idx[:patch_nums]]   # (topk, dim*2)

        L, dim = slide_tensor.shape
        if L > patch_nums:
            slide_tensor = slide_tensor[:patch_nums, :]
        elif L < patch_nums:
            pad_size = (0, 0, 0, patch_nums - L)  
            # pad_size 含义 (dim2_pad_left, dim2_pad_right, dim1_pad_left, dim1_pad_right)
            # 这里在第 0 维（序列长度）末尾补 (target_len - L) 行
            slide_tensor = F.pad(slide_tensor, pad_size, value=0)
    else:
        slide_tensor = torch.zeros(patch_nums, C_in)
    
    return slide_tensor

def gene_tSNE(save_prefix):
    slide_featlist,binary_labellist, mc_labellist = [],[],[]
    for row in tqdm(df_train.itertuples(index=False), total=len(df_train), ncols=80):
        slide_tensor = format_slide_tensor(f'{feat_dir}/{row.patientId}.pt', patch_nums, C_in)
        slide_featlist.append(slide_tensor)     # (L, dim) 
        binary_labellist.append(row.kfb_clsid)     # 0 or 1
        mc_labellist.append(classes.index(cls_map[row.kfb_clsname]))

    slide_featlist = torch.stack(slide_featlist)
    x_pool = slide_featlist.mean(dim=1)  # (B, C)
    x_pool_mc = torch.stack([feat for idx,feat in enumerate(slide_featlist) if mc_labellist[idx] != 0])
    x_pool_mc = x_pool_mc.mean(dim=1)[:, 512:]
    # t-SNE 降维
    tsne = TSNE(n_components=2, init='pca', random_state=42)
    x_2d_binary = tsne.fit_transform(x_pool.numpy())
    x_2d_mc = tsne.fit_transform(x_pool_mc.numpy())

    # ------------------------------
    # 二分类可视化
    plt.figure(figsize=(8, 6))
    binary_labellist = np.array(binary_labellist)

    colors = ['#1f77b4', '#ff7f0e']  # 蓝色和橙色
    for cls in [0, 1]:
        idx = binary_labellist == cls
        plt.scatter(
            x_2d_binary[idx, 0], x_2d_binary[idx, 1],
            c=colors[cls],
            label='Negative' if cls==0 else 'Positive',
            s=10,
            alpha=0.7
        )

    plt.legend(title='Binary Class')
    plt.title('t-SNE for Binary Classification')
    plt.savefig(f'{save_prefix}_binary_class.png', dpi=300)
    plt.close()

    # ------------------------------
    # 多分类可视化
    plt.figure(figsize=(8, 6))
    mc_labellist = np.array([i for i in mc_labellist if i != 0])

    hex_colors = ['#17becf', '#8c564b', '#2ca02c', '#9467bd', '#d62728']
    colors = np.array([mcolors.to_rgba(c) for c in hex_colors])

    for cls in set(mc_labellist):
        idx = mc_labellist == cls
        plt.scatter(
            x_2d_mc[idx, 0], x_2d_mc[idx, 1],
            color=colors[cls-1],
            label=classes[cls],
            s=10,
            alpha=0.7
        )

    plt.legend(title='Multi-Class')
    plt.title('t-SNE for Multi-Class Classification')
    plt.savefig(f'{save_prefix}_multi_class.png', dpi=300)
    plt.close()
        

if __name__ == "__main__":
    
    classes = ['NILM', 'AGC', 'ASC-US', 'LSIL', 'ASC-H', 'HSIL']
    cls_map = {
        'NILM': 'NILM',
        'ASC-US': 'ASC-US',
        'LSIL': 'LSIL',
        'ASC-H': 'ASC-H',
        'HSIL': 'HSIL',
        'SCC': 'HSIL',
        'AGC-N': 'AGC',
        'AGC': 'AGC',
        'AGC-NOS': 'AGC',
        'AGC-FN': 'AGC',
    }
    patch_nums,C_in = 400, 1024
    df_train = pd.read_csv('data_resource/0630/45_0924_train.csv')
    feat_dir = 'data_resource/0630/WINDOW_SIZE_1600/slide_feat_ours'  # mldecoder
    # feat_dir = 'data_resource/0630/WINDOW_SIZE_800/slide_feat_ours'
    save_dir = 'statistic_results/WSI_tSNE'
    os.makedirs(save_dir, exist_ok=True, mode=0o777)
    save_prefix = f'{save_dir}/mldecoder'
    # save_prefix = f'{save_dir}/wscernet'
    gene_tSNE(save_prefix)
