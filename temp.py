def save_wavelet_vis(xh_c, save_dir, prefix="wavelet", reduce="mean", 
                    scale_factor=2, contrast_enhance=True):
    """
    可视化 DTCWT 特征的六个方向 (幅值 + 相位)，支持通道维度聚合

    Args:
        xh_c: torch.Tensor, shape (C, 6, H, W, 2)
              单个样本的复小波系数
        save_dir: str, 保存文件夹路径
        prefix: str, 文件名前缀
        reduce: str, {"mean", "max"} 通道聚合方式
        scale_factor: int, 上采样倍数
        contrast_enhance: bool, 是否进行对比度增强
    """
    import os
    import torch
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    import numpy as np

    os.makedirs(save_dir, exist_ok=True)

    real = xh_c[..., 0]   # (C, 6, H, W)
    imag = xh_c[..., 1]   # (C, 6, H, W)

    # 幅值 (C, 6, H, W)
    magnitude = torch.sqrt(real**2 + imag**2)
    
    # 通道聚合
    if reduce == "mean":
        magnitude = magnitude.mean(0)  # (6, H, W)
    elif reduce == "max":
        magnitude = magnitude.max(0).values  # (6, H, W)
    else:
        raise ValueError(f"reduce must be 'mean' or 'max', got {reduce}")

    # 上采样 (使用双线性插值)
    if scale_factor > 1:
        # 添加批次和通道维度 (1, 1, 6, H, W) -> (1, 6, H, W)
        magnitude_upsampled = magnitude.unsqueeze(0)  # (1, 6, H, W)
        
        # 计算新的尺寸
        new_H = magnitude.shape[1] * scale_factor
        new_W = magnitude.shape[2] * scale_factor
        
        # 双线性插值上采样
        magnitude_upsampled = F.interpolate(
            magnitude_upsampled, 
            size=(new_H, new_W), 
            mode='bilinear', 
            align_corners=False
        )
        magnitude = magnitude_upsampled.squeeze(0)  # (6, H_new, W_new)

    # 转 numpy
    magnitude_np = magnitude.cpu().numpy()

    # 对比度增强
    if contrast_enhance:
        enhanced_magnitude = []
        for i in range(magnitude_np.shape[0]):
            # 对每个方向单独进行对比度增强
            img = magnitude_np[i]
            
            # 计算百分位数，避免极端值影响
            p2 = np.percentile(img, 2)
            p98 = np.percentile(img, 98)
            
            # 线性拉伸对比度
            img_enhanced = np.clip((img - p2) / (p98 - p2 + 1e-8), 0, 1)
            
            # Gamma 校正进一步增强对比度 (gamma < 1 提升亮部，>1 提升暗部)
            img_enhanced = img_enhanced ** 0.8
            
            enhanced_magnitude.append(img_enhanced)
        
        magnitude_np = np.array(enhanced_magnitude)

    # 保存幅值拼图
    fig, axes = plt.subplots(1, 6, figsize=(18, 3))
    for i in range(6):
        # 使用热力图颜色映射，调整显示范围
        im = axes[i].imshow(magnitude_np[i], cmap="hot", vmin=0, vmax=1)
        axes[i].set_title(f"Dir {i}")
        axes[i].axis("off")
    
    # 添加颜色条
    plt.tight_layout()
    plt.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    
    plt.savefig(os.path.join(save_dir, f"{prefix}_magnitude.png"), 
                bbox_inches='tight', dpi=300)
    plt.close()