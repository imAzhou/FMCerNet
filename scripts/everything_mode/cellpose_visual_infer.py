import json
from tqdm import tqdm
import torch
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from pycocotools.coco import COCO

def draw_boxes(img, bboxes, color=(0, 255, 0)):
    for box in bboxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    return img

def main():
    jsonfile = 'data_resource/HMCHH/annofiles_roi/fold1_train.json'
    img_root = 'data_resource/HMCHH/JPEGImages/'
    visual_saveroot = 'statistic_results/cellpose_infer/hmchh'
    os.makedirs(visual_saveroot, exist_ok=True, mode=0o777)
    with open(jsonfile, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    coco = COCO(jsonfile)
    
    for imgitem in tqdm(json_data['images'], ncols=80):
        purename = imgitem["file_name"].split('.')[0]
        img_id = imgitem['id']
        img_path = f'{img_root}/{imgitem["file_name"]}'
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gt_boxes = [ann['bbox'] for ann in coco.loadAnns(coco.getAnnIds(imgIds=img_id))]
        # 将xywh转为x1y1x2y2
        gt_boxes = [[x, y, x + w, y + h] for x, y, w, h in gt_boxes]
        
        preds = {}
        for dtag in ['d150']:
            predfile = f'data_resource/HMCHH/proposal_{dtag}/{purename}.json'
            preds[dtag] = []
            if os.path.exists(predfile):
                with open(predfile, 'r', encoding='utf-8') as f:
                    preds[dtag] = json.load(f)
        
        fig, axs = plt.subplots(1, 2, figsize=(12, 7))
        titles = ['GT', 'd150']
        imgs = [
            draw_boxes(img.copy(), gt_boxes, color=(255, 0, 0)),
            draw_boxes(img.copy(), preds['d150'], color=(0, 255, 0)) if preds['d150'] else img.copy(),
            # draw_boxes(img.copy(), preds['d180'], color=(0, 0, 255)) if preds['d180'] else img.copy(),
        ]
        for ax, im, title in zip(axs, imgs, titles):
            ax.imshow(im)
            ax.set_title(title)
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(f'{visual_saveroot}/{purename}.png')
        plt.close()

def find_imgitem(purename, json_data):
    for imgitem in json_data['images']:
        if imgitem['file_name'] == f'{purename}.png':
            return imgitem

def find_imganns(imgitem, json_data):
    imgbboxes = []
    for annitem in json_data['annotations']:
        if annitem['image_id'] == imgitem['id']:
            x1,y1,w,h = annitem['bbox']
            imgbboxes.append([x1, y1, x1+w, y1+h])
    return imgbboxes

def test_demo_diff_d():
    from cellpose import models,utils
    model = models.CellposeModel(gpu=True, 
                                #  pretrained_model='/x22201018/.cellpose/models/cpsam',
                                 device=torch.device("cuda:1"))
    diameters = [30, 60, 90, 120, 150, 180, 210, 240]
    with open('data_resource/HMCHH/annofiles_roi/fold1_train.json', 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    purenames = ['1657bj008_0001','1657bj008_0096']
    for purename in purenames:
        img_url = f'data_resource/HMCHH/JPEGImages/{purename}.png'
        imgitem = find_imgitem(purename, json_data)
        gt_bboxes = find_imganns(imgitem, json_data)
        img = cv2.imread(img_url)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        total_bboxes, total_outXY = [],[]
        for dia in diameters:
            masks_pred, flows, styles = model.eval([img], 
                                            # niter=4000,     # 根据 cellprob & 预测距离计算 cell 需要的迭代次数
                                            batch_size=64,
                                            max_size_fraction=1,
                                            diameter=float(dia)) # using more iterations for bacteria
            masks_pred = masks_pred[0]
            # 输出每个区域的外包围框坐标 (x1, y1, x2, y2)
            bboxes = []
            labels = np.unique(masks_pred)
            labels = labels[labels != 0]  # 排除背景 label 0
            for label in labels:
                ys, xs = np.where(masks_pred == label)
                y1, y2 = ys.min(), ys.max()
                x1, x2 = xs.min(), xs.max()
                bboxes.append([x1, y1, x2 + 1, y2 + 1])  # 注意：右边是非包含式，+1 保持一致性
            bboxes = np.array(bboxes).tolist()
            total_bboxes.append(bboxes)
            outlines = utils.masks_to_outlines(masks_pred)
            outX, outY = np.nonzero(outlines)
            total_outXY.append((outX, outY))
        
        fig, axs = plt.subplots(3, 3, figsize=(12, 12))
        axs = axs.flatten()
        im = draw_boxes(img.copy(), gt_bboxes, color=(255, 0, 0))
        axs[0].imshow(im)
        axs[0].set_title('GT')
        axs[0].axis('off')

        for idx in range(1,9):
            bboxes = total_bboxes[idx-1]
            outX, outY = total_outXY[idx-1]
            im = draw_boxes(img.copy(), bboxes, color=(0, 255, 0)) if bboxes else img.copy()
            im[outX, outY] = np.array([0, 255, 0])
            axs[idx].imshow(im)
            axs[idx].set_title(f'd{diameters[idx-1]}')
            axs[idx].axis('off')

        plt.tight_layout()
        visual_saveroot = 'statistic_results/cellpose_infer/hmchh_demo'
        os.makedirs(visual_saveroot, exist_ok=True, mode=0o777)
        plt.savefig(f'{visual_saveroot}/{purename}.png')
        plt.close()

def visual_demo():
    from cellpose import models,utils,transforms,dynamics
    
    def flow_to_cell_prob(dP):
        """Convert flow field to cell probability map.
        
        Args:
            dP (ndarray): Flow field [dy, dx], shape (2, H, W)
        
        Returns:
            ndarray: Cell probability map, shape (H, W), values in [0, 1]
        """
        # 计算每个像素的光流模长（即 flow 强度）
        magnitude = np.sqrt(np.sum(dP**2, axis=0))
        # 使用 99% 分位归一化增强对比
        norm_mag = transforms.normalize99(magnitude)
        # 限制在 0~1
        prob_map = np.clip(norm_mag, 0, 1)
        prob_map = ndimage.gaussian_filter(prob_map, sigma=1.0)
             
        # 计算散度 (divergence)
        # 使用中心差分计算梯度
        divergence = np.gradient(dP[0], axis=0) + np.gradient(dP[1], axis=1)
        # 对散度阈值化
        boundary_mask = divergence > np.percentile(np.abs(divergence), 90)
        
        return prob_map,boundary_mask

    cellpose_model = models.CellposeModel(gpu=True, 
                                #  pretrained_model='/x22201018/.cellpose/models/cpsam',
                                 device=torch.device("cuda:1"))
    
    
    diameters,flow_thresholds,cellprobThr = [15, 30, 120],[0.6,0.6,0.8],[0.2,0.2,0.1]
    with open('data_resource/HMCHH/annofiles_roi/fold1_train.json', 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    purenames = ['1662bj013_0096','1657bj008_0242']
    for purename in purenames:
        img_url = f'data_resource/HMCHH/JPEGImages/{purename}.png'
        imgitem = find_imgitem(purename, json_data)
        gt_bboxes = find_imganns(imgitem, json_data)
        img = cv2.imread(img_url)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        fig, axs = plt.subplots(3, 3, figsize=(12, 12))
        axs = axs.flatten()

        idx_current = 0
        for dia,ft,ct in zip(diameters,flow_thresholds,cellprobThr):
            masks_pred, results, styles = cellpose_model.eval(
                [img], batch_size=64, flow_threshold=ft, diameter=dia, 
                compute_masks=False, augment=True, resample=True)

            flowi, dP, cellprob = results[0]
            cellprob,boundary_mask = flow_to_cell_prob(dP)
            cellprob[boundary_mask] = 0.
            maski = dynamics.resize_and_compute_masks(
                    dP, cellprob,
                    cellprob_threshold=ct,
                    flow_threshold=ft, resize=None,
                    min_size=15, max_size_fraction=0.7,
                    device=cellpose_model.device)
            
            outlines = utils.masks_to_outlines(maski).astype(np.uint8)
            contours, _ = cv2.findContours(outlines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            im = img.copy()
            cv2.drawContours(im, contours, -1, color=(0, 255, 0), thickness=2)
            axs[idx_current].imshow(im)
            axs[idx_current].set_title(f'd{dia}-outlines')
            axs[idx_current].axis('off')
            idx_current += 1

            axs[idx_current].imshow(flowi)
            axs[idx_current].set_title(f'd{dia}-flow')
            axs[idx_current].axis("off")
            idx_current += 1

            axs[idx_current].imshow(cellprob, cmap='gray', vmin=0, vmax=1)
            axs[idx_current].set_title(f'd{dia}-cell prob')
            axs[idx_current].axis('off')
            idx_current += 1

        plt.tight_layout()
        visual_saveroot = 'statistic_results/cellpose_infer/hmchh_demo/0722'
        os.makedirs(visual_saveroot, exist_ok=True, mode=0o777)
        plt.savefig(f'{visual_saveroot}/{purename}.png')
        plt.close()

def smooth_instance_mask(instance_mask: np.ndarray, keep_ratio: float = 0.01) -> np.ndarray:
    """
    使用 NumPy FFT 实现傅里叶描述子平滑每个细胞实例轮廓。

    Args:
        instance_mask (np.ndarray): 输入实例 mask，0 表示背景，1~k 表示不同实例。
        keep_ratio (float): 保留的低频比例（0.01 ~ 0.2，越小越光滑）

    Returns:
        np.ndarray: 平滑后的实例 mask。
    """
    smoothed_mask = np.zeros_like(instance_mask, dtype=np.uint16)
    instance_ids = np.unique(instance_mask)
    instance_ids = instance_ids[instance_ids != 0]  # 排除背景

    current_id = 1
    for inst_id in instance_ids:
        binary_mask = (instance_mask == inst_id).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) == 0:
            continue

        for contour in contours:
            if contour.shape[0] < 10:
                continue  # 太短的边界跳过

            contour_points = contour[:, 0, :]  # (N, 2)
            N = len(contour_points)
            z = contour_points[:, 0] + 1j * contour_points[:, 1]  # 转为复数表示轮廓

            Z = np.fft.fft(z)

            # 保留低频，去除高频（左右对称）
            K = max(1, int(N * keep_ratio))
            Z_filtered = np.zeros_like(Z)
            Z_filtered[:K] = Z[:K]
            Z_filtered[-K:] = Z[-K:]

            z_smooth = np.fft.ifft(Z_filtered)
            smoothed_contour = np.stack([np.real(z_smooth), np.imag(z_smooth)], axis=1)
            smoothed_contour = np.round(smoothed_contour).astype(np.int32)

            # 防止超出边界
            smoothed_contour[:, 0] = np.clip(smoothed_contour[:, 0], 0, instance_mask.shape[1] - 1)
            smoothed_contour[:, 1] = np.clip(smoothed_contour[:, 1], 0, instance_mask.shape[0] - 1)

            # 绘制平滑轮廓
            smoothed_contour = smoothed_contour.reshape(-1, 1, 2)
            cv2.drawContours(smoothed_mask, [smoothed_contour], -1, int(current_id), thickness=-1)
            current_id += 1

    return smoothed_mask


if __name__ == "__main__":
    # main()
    # test_demo_diff_d()    # 可视化不同细胞直径的推理结果（仅绘制矩形框）
    visual_demo()    # 可视化指定细胞直径的推理结果（绘制细胞边界及距离光流图）
