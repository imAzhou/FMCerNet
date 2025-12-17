from tqdm import tqdm
from cerwsi.nets import CellposeNet
from cellpose import models,utils,dynamics
import torch
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from pycocotools import mask as mask_utils
from cerwsi.utils import inst2bboxes,flow2cellprob
import random
from glob import glob

tile_test_bs = 128
cell_config = {
    # 'nucleus': dict(dia=15, flowThr=0.6, cellprobThr=0.1, min_size=15),
    # 'cytoplasm': dict(dia=120, flowThr=0.8, cellprobThr=0.1, min_size=10*10),
    'cluster': dict(dia=240, flowThr=-1, cellprobThr=0.1, min_size=30*30),
}
tail = 'cluster'
device = torch.device('cuda:7')
cellpose_ckpt = 'checkpoints/cpsam'

# all_CDetector = glob('/c23030/zly/datasets/CervicalDatasets/ComparisonDetectorDataset/train/*.bmp')
# random.shuffle(all_CDetector)
# all_HMCHH = glob('/c23030/zly/datasets/CervicalDatasets/HMCHH/JPEGImages/*.png')
# random.shuffle(all_HMCHH)
# tag_imgs = [*all_CDetector[:20], *all_HMCHH[:20]]

tag_imgs = []
for vispath in glob('statistic_results/cellpose_infer/contrast/*_cytoplasm.png'):
    purename = os.path.basename(vispath).replace('_cytoplasm.png', '')
    if len(purename) == 5:
        imgpath = f'/c23030/zly/datasets/CervicalDatasets/ComparisonDetectorDataset/train/{purename}.bmp'
    else:
        imgpath = f'/c23030/zly/datasets/CervicalDatasets/HMCHH/JPEGImages/{purename}.png'
    tag_imgs.append(imgpath)


def infer_single_img(img_RGB, cellpose_model):
    mask_instlist = []
    for ctype,config in cell_config.items():
        dia = float(config['dia'])
        masks_pred, results, styles = cellpose_model.eval([img_RGB], batch_size=64, 
            flow_threshold=0.4, diameter=dia, augment=True, compute_masks=True)
        objects = inst2bboxes(masks_pred[0], userle=True)
        mask_instlist.extend(objects)
    return mask_instlist

def smooth_instance_mask(mask: np.ndarray, keep_ratio: float = 0.05) -> np.ndarray:
    """
    对单个二值 mask 进行傅里叶轮廓平滑，
    keep_ratio 越小 → 模糊越强（形状更圆滑），越大 → 模糊越弱（保留更多细节）。
    """
    smoothed_mask = np.zeros(mask.shape, dtype=np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return mask

    for contour in contours:
        if contour.shape[0] < 10:
            continue

        contour_points = contour[:, 0, :]
        N = len(contour_points)
        z = contour_points[:, 0] + 1j * contour_points[:, 1]
        Z = np.fft.fft(z)

        K = max(1, int(N * keep_ratio))
        Z_filtered = np.zeros_like(Z)
        Z_filtered[:K] = Z[:K]
        Z_filtered[-K:] = Z[-K:]

        z_smooth = np.fft.ifft(Z_filtered)
        smoothed_contour = np.stack([np.real(z_smooth), np.imag(z_smooth)], axis=1)
        smoothed_contour = np.round(smoothed_contour).astype(np.int32)

        # 防止越界
        smoothed_contour[:, 0] = np.clip(smoothed_contour[:, 0], 0, mask.shape[1] - 1)
        smoothed_contour[:, 1] = np.clip(smoothed_contour[:, 1], 0, mask.shape[0] - 1)

        if smoothed_contour.shape[0] < 3:
            continue  # 至少要3个点才能绘制封闭轮廓

        smoothed_contour = smoothed_contour.reshape(-1, 1, 2).astype(np.int32)
        cv2.drawContours(smoothed_mask, [smoothed_contour], -1, 1, thickness=-1)

    return smoothed_mask

def visualize_segmentation_comparison(imgRGB, our_object_list, cellpose_object_list, save_path):
    """
    可视化原图与两组分割结果的比较。
    
    Args:
        imgpath (str): 原图路径。
        our_object_list (list[dict]): 第一组检测结果，每个元素包含 'segmentation'。
        cellpose_object_list (list[dict]): 第二组检测结果。
        save_path (str): 输出图片保存路径。
    """
    # 创建两张副本用于绘制边缘
    img_our = imgRGB.copy()
    img_cellpose = imgRGB.copy()

    def draw_segmentation_edges(target_img, object_list, smooth=False):
        """绘制 segmentation 的边缘线，可选择平滑。"""
        for obj in object_list:
            rle = obj['segmentation']
            mask = mask_utils.decode(rle).astype(np.uint8)

            if smooth:
                mask = smooth_instance_mask(mask)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
            cv2.drawContours(target_img, contours, -1, color, 4)

    # 平滑 our_object_list 再绘制
    draw_segmentation_edges(img_our, our_object_list, smooth=True)
    # 原样绘制 cellpose
    draw_segmentation_edges(img_cellpose, cellpose_object_list, smooth=False)

    # 绘制三行图像
    fig, axes = plt.subplots(3, 1, figsize=(8, 12))
    axes[0].imshow(imgRGB)
    axes[0].set_title("Original Image")
    axes[1].imshow(img_our)
    axes[1].set_title("Our Model Segmentation")
    axes[2].imshow(img_cellpose)
    axes[2].set_title("Cellpose Segmentation")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"可视化结果已保存到：{save_path}")
    

def main():
    save_dir = 'statistic_results/cellpose_infer/contrast'
    os.makedirs(save_dir, exist_ok=True, mode=0o777)

    our_model = CellposeNet(cell_config, userle=True).to(device)
    our_model.load_ckpt(cellpose_ckpt)
    cellpose_model = models.CellposeModel(
        gpu=True, pretrained_model=cellpose_ckpt, device=device)
    
    for imgpath in tqdm(tag_imgs, ncols=80):
        filename = os.path.basename(imgpath)
        purename = filename.split('.')[0]
        img = cv2.imread(imgpath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        our_object_list = our_model(img, batchsize=tile_test_bs)
        cellpose_object_list = infer_single_img(img, cellpose_model)

        visualize_segmentation_comparison(
            img, our_object_list, cellpose_object_list, f'{save_dir}/{purename}_{tail}.png')

def draw_flow():
    save_dir = 'statistic_results/cellpose_infer/flow_map'
    os.makedirs(save_dir, exist_ok=True, mode=0o777)
    cellpose_model = models.CellposeModel(
        gpu=True, pretrained_model=cellpose_ckpt, device=device)
    
    imgpath = '/c23030/zly/datasets/CervicalDatasets/ComparisonDetectorDataset/train/00813.bmp'
    filename = os.path.basename(imgpath)
    purename = filename.split('.')[0]
    img = cv2.imread(imgpath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for dia in [15.0, 120.0, 240.0]:
        masks_pred, results, styles = cellpose_model.eval([img], batch_size=64, 
            flow_threshold=0.4, diameter=dia, augment=True, compute_masks=True)
        flowi, dP, cellprob = results[0]
        new_cellprob,boundary_mask = flow2cellprob(dP)
        # new_cellprob[boundary_mask] = 0.

        fig, axs = plt.subplots(3, 2, figsize=(12, 8))
        axs = axs.flatten()

        axs[0].imshow(img)
        axs[1].imshow(flowi)
        axs[2].imshow(cellprob, cmap='gray', vmin=0, vmax=1)
        axs[2].set_title("Cellpose cellprob")
        axs[3].imshow(boundary_mask, cmap='gray', vmin=0, vmax=1)
        axs[3].set_title("Our boundary_mask")
        axs[4].imshow(new_cellprob, cmap='gray', vmin=0, vmax=1)
        axs[4].set_title("Our new_cellprob")

        for ax in axs:
            ax.axis("off")
        
        save_path = f'{save_dir}/{purename}_d{int(dia)}.png'
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"可视化结果已保存到：{save_path}")

def crop_tgt():
    tgt_dir = 'statistic_results/cellpose_infer/00813'
    crop_savedir = f'{tgt_dir}_crop'
    os.makedirs(crop_savedir, exist_ok=True, mode=0o777)

    # 遍历目标文件夹内的所有图片
    for filename in os.listdir(tgt_dir):
        img_path = os.path.join(tgt_dir, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: cannot read image {img_path}")
            continue

        h, w = img.shape[:2]

        # 裁剪右上角 1/4 区域
        crop_img = img[0:h//2, w//2:w]

        # 保存到新目录
        save_path = os.path.join(crop_savedir, filename)
        cv2.imwrite(save_path, crop_img)

        print(f"Saved cropped image to: {save_path}")

if __name__ == "__main__":
    # main()
    # draw_flow()
    crop_tgt()
