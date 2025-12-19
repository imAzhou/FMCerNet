import json
from torchvision.ops import nms
import torch
import numpy as np
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from cerwsi.nets.cellpose import transforms
import matplotlib.patches as patches
from collections import defaultdict
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def flow2cellprob(dP):
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

def gene_gridboxes(H, W, grid_size):
    xs = torch.arange(0, W, grid_size)
    ys = torch.arange(0, H, grid_size)

    # 网格起点坐标（左上角）
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    x1 = xx.flatten()
    y1 = yy.flatten()

    # 右下角坐标裁剪至图像边界
    x2 = (x1 + grid_size).clamp(max=W)
    y2 = (y1 + grid_size).clamp(max=H)

    # 过滤掉无效框（宽或高为0）
    valid = (x2 > x1) & (y2 > y1)

    grid_boxes = torch.stack([x1[valid], y1[valid], x2[valid], y2[valid]], dim=1)
    return grid_boxes

def postprocess_bboxes(bboxes, grid_boxes, maxdet, minlen=0):
    """
    后处理bbox列表，使其数量为 maxdet，按规则调整score和筛选。

    Args:
        bboxes (list): shape (N, 4)，格式为 (x1, y1, x2, y2)，初始 score 均为 1.0。
        grid_boxes (Tensor): 网格化生成的候选框
        maxdet (int): 保留的 bbox 数量
        minlen (int): 保留 w>minlen & h>minlen 的bbox

    Returns:
        boxes: Tensor: shape (maxdet, 4)，格式为 (x1, y1, x2, y2)
        scores: Tensor: shape (maxdet, )
    """
    iou_threshold = 0.3
    bboxes = torch.tensor(bboxes, dtype=torch.float32)  # (x1, y1, x2, y2)
    wh = bboxes[:, 2:4] - bboxes[:, 0:2]  # 计算宽和高，shape = (N, 2)
    keep = (wh[:, 0] > minlen) & (wh[:, 1] > minlen)  # 同时满足宽高都大于 minlen
    bboxes = bboxes[keep]
    
    # 先按 nfs 去重
    scores = torch.ones((bboxes.shape[0],))
    keep = nms(bboxes, scores, iou_threshold=iou_threshold)
    bboxes = bboxes[keep]
    
    N = bboxes.shape[0]
    if N == maxdet:
        scores = torch.ones((N,))
        return bboxes, scores

    elif N < maxdet:
        num_needed = maxdet - N
        grid_scores = torch.full((grid_boxes.shape[0],), 0.5, dtype=torch.float32)
        extra = torch.randperm(len(grid_scores))[:num_needed]
        orig_scores = torch.ones((N,))
        all_boxes = torch.cat([bboxes, grid_boxes[extra]], dim=0)
        all_scores = torch.cat([orig_scores, grid_scores[extra]], dim=0)

        keep = nms(all_boxes, all_scores, iou_threshold=iou_threshold)
        if len(keep) >= maxdet:
            selected = keep[:maxdet]
        else:
            # 从未被保留的索引中随机补充
            all_indices = torch.arange(all_boxes.shape[0], device=all_boxes.device)
            unkept = all_indices[~torch.isin(all_indices, keep)]
            num_needed = maxdet - len(keep)
            extra = unkept[torch.randperm(len(unkept))[:num_needed]]
            selected = torch.cat([keep, extra], dim=0)

        return all_boxes[selected], all_scores[selected]

    else:  # N > maxdet
        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        sorted_idx = torch.argsort(areas, descending=True)
        bboxes = bboxes[sorted_idx]
        x1 = bboxes[:, 0].view(N, 1)
        y1 = bboxes[:, 1].view(N, 1)
        x2 = bboxes[:, 2].view(N, 1)
        y2 = bboxes[:, 3].view(N, 1)

        prev_x1 = bboxes[:, 0].view(1, N)
        prev_y1 = bboxes[:, 1].view(1, N)
        prev_x2 = bboxes[:, 2].view(1, N)
        prev_y2 = bboxes[:, 3].view(1, N)

        tolerance = 5
        inside = (
            (x1 >= prev_x1 - tolerance) &
            (y1 >= prev_y1 - tolerance) &
            (x2 <= prev_x2 + tolerance) &
            (y2 <= prev_y2 + tolerance)
        )  # shape: (N, N)

        # 只看前面的框（面积更大）
        mask = torch.triu(torch.ones(N, N, dtype=torch.bool, device=bboxes.device), diagonal=1)
        inside = inside & mask

        # 对每个框，如果有任何前面的框包含它 → 被包含
        is_contained = inside.any(dim=1)
        scores = torch.where(is_contained, torch.tensor(0.5), torch.tensor(1.0))
        keep = nms(bboxes, scores, iou_threshold=iou_threshold)
        selected = keep[:maxdet]
        final_bboxes,final_scores = bboxes[selected], scores[selected]
        if len(keep) < maxdet:
            num_needed = maxdet - len(keep)
            grid_scores = torch.full((grid_boxes.shape[0],), 0.5, dtype=torch.float32)
            extra = torch.randperm(len(grid_scores))[:num_needed]
            final_bboxes = torch.cat([final_bboxes, grid_boxes[extra]], dim=0)
            final_scores = torch.cat([final_scores, grid_scores[extra]], dim=0)
        return final_bboxes,final_scores

def inst2bboxes(instmask, userle = False):
    H,W = instmask.shape
    bboxes_list = []
    slices = ndimage.find_objects(instmask)
    for instid, slc in enumerate(slices, start=1):
        y1, y2 = max(0, slc[0].start), min(H, slc[0].stop)
        x1, x2 = max(0, slc[1].start), min(W, slc[1].stop)
        w, h = x2 - x1, y2 - y1
        bboxItem = {
            "bbox": [x1,y1,x2,y2],
            'cxcy':[x1 + w/2, y1 + h/2],
            'w': w,
            'h': h,
            'area': w*h,
        }
        if userle:
            rle = mask_utils.encode(np.asfortranarray(instmask==instid))
            rle['counts'] = rle['counts'].decode('utf-8')
            bboxItem['segmentation'] = rle
        bboxes_list.append(bboxItem)
    return bboxes_list

def missed_visualize_fn(coco_gt, pred_info, missed_anns, image_root, visual_saveroot, num_images=5, maxDet=1000):
    """
    可视化未召回 ann 所在图像（GT vs 预测框）
    
    参数:
        coco_gt: COCO ground truth 对象
        pred_info: list[dict], 预测结果，每个包含 image_id, bbox, score, category_id
        missed_anns: list[dict], 未被召回的 anns
        image_root: 图像路径根目录
        num_images: 可视化的图像数量（默认最多显示 5 张）
    """

    # 收集每张图的未召回 anns
    imgid_to_missed = defaultdict(list)
    for ann in missed_anns:
        imgid_to_missed[ann['image_id']].append(ann)

    # 转换预测数据为 image_id -> list[bbox]
    imgid_to_preds = defaultdict(list)
    for pred in pred_info:
        imgid_to_preds[pred['image_id']].append(pred)

    # 只显示前 num_images 个图像
    for i, (img_id, missed_list) in enumerate(imgid_to_missed.items()):
        if i >= num_images:
            break
        img_info = coco_gt.loadImgs([img_id])[0]
        img_path = os.path.join(image_root, img_info['file_name'])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 创建图
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].imshow(img)
        axs[0].set_title(f"GT (missed anns)")
        axs[1].imshow(img)
        axs[1].set_title(f"Predicted bboxes")

        # 画GT中未召回的框（红色）
        for ann in missed_list:
            x, y, w, h = ann['bbox']
            axs[0].add_patch(patches.Rectangle((x, y), w, h, edgecolor='red', facecolor='none', linewidth=2))
            axs[0].text(
                x, y - 2,  # 往上偏移一点
                f'maxiou: {ann["max_iou"]:.2f}',
                fontsize=8,
                color='red',
                verticalalignment='top',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1)
            )

        # 画预测框（蓝色）
        cnt = 0
        for pred in imgid_to_preds[img_id]:
            if cnt == maxDet:
                break
            x, y, w, h = pred['bbox']
            axs[1].add_patch(patches.Rectangle((x, y), w, h, edgecolor='blue', facecolor='none', linewidth=1))
            cnt += 1
        
        for ax in axs:
            ax.axis('off')
        plt.tight_layout()
        
        plt.savefig(f"{visual_saveroot}/{img_info['file_name']}")
        plt.close()

def missed_analyze(gt_jsonfile, proposal_jsonfile, iou_thr=0.3, maxDet=1000, 
                   vis_num_images=50, visual_saveroot='', image_root=''):
    '''
    Args:
        gt_jsonfile: gt json path
        proposal_jsonfile: proposal json path
        iou_thr: 只评估 IoU<iou_thr 的情况
        maxDet: 只评估 最大候选框数量=maxDet 的情况
        num_images: 可视化漏检样本数量（以图片为单位）
        visual_saveroot: 可视化结果保存的文件夹路径
        image_root: 图片所在文件夹的路径
    '''

    coco_gt = COCO(gt_jsonfile)
    with open(proposal_jsonfile, 'r', encoding='utf-8') as f:
        pred_info = json.load(f)
    coco_dt = coco_gt.loadRes(pred_info)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.params.iouThrs = [iou_thr]
    coco_eval.params.maxDets = [maxDet]
    coco_eval.evaluate()
    coco_eval.accumulate()

    # 获取每张图像、每个类别的未召回 GT anns 和对应的最大 IoU
    missed_anns = []
    for eval_img in coco_eval.evalImgs:
        if eval_img is None:
            continue
        if eval_img['aRng'][1] < 10000: # 只考虑不分面积大小的匹配
            continue

        gt_ids = eval_img['gtIds']  # 当前 image + category 的所有 GT ann ids
        dt_ids = eval_img['dtIds']  # 检测到的 ann ids
        gt_matches = eval_img['gtMatches'][0]  # IoU=0.3 时每个 gt 的匹配情况
        image_id = eval_img['image_id']
        # imginfo = coco_gt.loadImgs([image_id])[0]
        # if imginfo['file_name'] != '1657bj008_0215.png':
        #     continue

        category_id = eval_img['category_id']
        # 用 (image_id, category_id) 作为 key 从 coco_eval.ious 获取 iou 矩阵
        ious_this = coco_eval.ious.get((image_id, category_id)).T  # shape: [num_gt, num_dt]

        for idx, gt_id in enumerate(gt_ids):
            if gt_matches[idx] == 0:  # 没有匹配成功的 dt
                # 该 gt 对应的所有 dt 的 IoU
                ious_for_this_gt = ious_this[idx]
                max_iou = np.max(ious_for_this_gt) if len(ious_for_this_gt) else 0.0
                ann = coco_gt.loadAnns([gt_id])[0]
                ann['max_iou'] = max_iou
                missed_anns.append(ann)

    missed_visualize_fn(
        coco_gt=coco_gt,
        pred_info=pred_info,
        missed_anns=missed_anns,
        image_root=image_root,
        visual_saveroot=visual_saveroot,
        num_images=vis_num_images,
        maxDet = maxDet
    )
    
    # 分析宽高
    bbox_wh = [(ann['bbox'][2], ann['bbox'][3]) for ann in missed_anns]
    widths, heights = zip(*bbox_wh) if bbox_wh else ([], [])
    ann_ids = coco_gt.getAnnIds()
    anns = coco_gt.loadAnns(ann_ids)
    print(f"未召回 ann 数量: {len(bbox_wh)}/{len(anns)}")
    if bbox_wh:
        print(f"平均宽度: {np.mean(widths):.2f}, 平均高度: {np.mean(heights):.2f}")
        print(f"中位宽度: {np.median(widths):.2f}, 中位高度: {np.median(heights):.2f}")
    else:
        print("所有 ann 都被召回。")
