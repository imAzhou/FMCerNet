import json
import torch
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.ops import nms
from torchvision import transforms as T
from scipy import ndimage
# from cellpose import models,utils,transforms,dynamics
from pycocotools import mask as mask_utils
from mmdet.evaluation import CocoMetric
from scipy.spatial import ConvexHull
from tqdm import tqdm


def visual_scored_masks(orig_image, annlist, primary_metric, save_path, topk=100, minsize=30*30, mode='top'):
    """
    Args:
        orig_image (np.ndarray): 原图 (H,W,3)，用于裁切可视化
        annlist (List[Dict]): 来自 format_mask 的 masklist，每项包含 segmentation, bbox, scores
        save_path (str): 最终保存图片路径
    """
    # 1. 过滤面积大于 minsize 且按 final_score 排序
    valid_anns = [
        ann for ann in annlist
        if ann["area"] > minsize
    ]
    reverse = mode == 'top'  # 'top' 排序方向为高->低；'bottom' 为低->高
    valid_anns.sort(key=lambda x: x['scores'][primary_metric], reverse=reverse)
    top_anns = valid_anns[:topk]

    # 2. 裁切图像、绘制轮廓、resize并填充
    def resize_and_pad(img, size=(100, 100)):
        h, w = img.shape[:2]
        scale = min(size[0] / h, size[1] / w)
        nh, nw = int(h * scale), int(w * scale)
        img_resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.ones((size[1], size[0], 3), dtype=np.uint8) * 255
        dy, dx = (size[1] - nh) // 2, (size[0] - nw) // 2
        canvas[dy:dy+nh, dx:dx+nw] = img_resized
        return canvas

    vis_results = []
    for ann in top_anns:
        x1, y1, w, h = ann["bbox"]
        x1, y1, x2, y2 = int(x1), int(y1), int(x1 + w), int(y1 + h)
        crop_img = orig_image[y1:y2, x1:x2].copy()
        
        # 解码 rle 得到 mask & 获取边界轮廓
        ann_mask = mask_utils.decode(ann["segmentation"]).astype(np.uint8) * 255
        contours, _ = cv2.findContours(ann_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_shifted = [cnt - np.array([[x1, y1]]) for cnt in contours]
        maxlen = max(w,h)
        thickness = 1 if maxlen<100 else 2
        cv2.drawContours(crop_img, contours_shifted, -1, (0, 255, 0), thickness=thickness)

        # resize and pad
        resized = resize_and_pad(crop_img)
        vis_results.append(resized)

    # 3. 拼成10x10网格展示
    grid_size = 10
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(30, 30))
    for idx, (img, ann) in enumerate(zip(vis_results, top_anns)):
        row, col = divmod(idx, grid_size)
        axs[row, col].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        title = f'{primary_metric}: {ann["scores"][primary_metric]:.2f}'
        axs[row, col].set_title(title)
        axs[row, col].axis('off')

    # 清除多余子图（若 topk < 100）
    for idx in range(len(top_anns), grid_size * grid_size):
        row, col = divmod(idx, grid_size)
        axs[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def visual_merged_masks(img, merged_mask, save_path, size_thresh=30):
    color_map = {
        "nucleus": (0, 255, 0),     # Green
        "cytoplasm": (0, 255, 255),   # Yellow
        "cluster": (0, 0, 255),     # Red
    }
    img = img.copy()  # 避免原图被改

    for mask_info in merged_mask:
        x, y, w, h = mask_info['bbox']
        ctype = mask_info["ctype"]
        color = color_map.get(ctype, (128, 128, 128))
        if w < size_thresh and h < size_thresh:
            rle = mask_info["segmentation"]
            m = mask_utils.decode(rle).astype(np.uint8)
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours, -1, color, thickness=2)
        else:
            pt1 = (int(x), int(y))
            pt2 = (int(x + w), int(y + h))
            cv2.rectangle(img, pt1, pt2, color, thickness=2)

    # 显示图像
    fig, axs = plt.subplots(1, 1, figsize=(6, 6))
    axs.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axs.set_title("Merged Masks in BBoxes")
    axs.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def postprocess(pred_mask, pred_boundary):
    """
    输入:
        pred_mask: numpy数组 (h, w), 布尔型掩码 (True=前景, False=背景)
        pred_boundary: numpy数组 (h, w), 布尔型边界掩码 (True=边界, False=非边界)
    
    返回:
        pred_instmask: 实例分割掩码 (0: 背景, 1-N: 实例ID)
    """
    pred_mask = T.ToPILImage()(pred_mask).convert('RGB')
    pred_mask = cv2.cvtColor(np.asarray(pred_mask),cv2.COLOR_RGB2GRAY)
    _,pred_mask = cv2.threshold(pred_mask,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    pred_boundary = T.ToPILImage()(pred_boundary).convert('RGB')
    pred_boundary = cv2.cvtColor(np.asarray(pred_boundary),cv2.COLOR_RGB2GRAY)  
    pred_boundary = cv2.normalize(pred_boundary, dst=None, alpha=350, beta=10, norm_type=cv2.NORM_MINMAX)
    _,pred_boundary = cv2.threshold(pred_boundary, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        
    
    h, w = pred_mask.shape

    # 步骤1: 将布尔掩码转为 uint8 (0/255)
    # mask_uint8 = np.where(pred_mask, 255, 0).astype(np.uint8)  # True=255, False=0
    # boundary_uint8 = np.where(pred_boundary, 255, 0).astype(np.uint8)  # True=255, False=0

    # # 步骤2: 获取轮廓（掩码与非边界的交集）
    # non_boundary = cv2.bitwise_not(boundary_uint8)  # 反转边界（边界=0，非边界=255）
    # pred_contours = cv2.bitwise_and(mask_uint8, non_boundary)  # 掩码 ∩ 非边界区域

    # 步骤3: 形态学处理（清理噪声）
    pred_contours = cv2.bitwise_and(pred_boundary, pred_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3, 3)) 
    pred_contours = cv2.erode(pred_contours, kernel, iterations=1)
    pred_contours = cv2.dilate(pred_contours, kernel, iterations=1)

    # 步骤4: 查找轮廓并过滤小面积区域
    contours, _ = cv2.findContours(pred_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    markers = np.zeros((h, w), dtype=np.int32)  # 初始化标记图

    if len(contours) > 0:
        areas = [cv2.contourArea(cnt) for cnt in contours]
        min_area = np.mean(areas) / 5  # 过滤掉面积小于均值1/5的轮廓

        # 为每个有效轮廓分配唯一ID（从1开始）
        for i, cnt in enumerate(contours):
            if cv2.contourArea(cnt) > min_area:
                cv2.drawContours(markers, [cnt], -1, i+1, -1)  # 填充轮廓为i+1

    # 步骤5: 执行分水岭算法
    mask_3ch = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2BGR)  # 转为3通道（cv2.watershed要求）
    cv2.watershed(mask_3ch, markers)

    # 后处理：边界标记为-1，归为背景0
    pred_instmask = np.where(markers == -1, 0, markers)

    return pred_instmask

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

def old_mask_score_fn(inst_mask,
        score_ratio=(0.5, 0.5, 0.0),
        compactness_range=(4 * np.pi, 200),
        dispersion_suppression_thresh=2.0,
        dispersion_scale=1.0):
    """
    Compute shape quality scores for a binary instance mask.

    Args:
        inst_mask (ndarray): 2D bool or 0/1 array, instance mask.
        weights (tuple): Weights for (compactness, solidity, centroid_dispersion).
        compactness_range (tuple): (min, max) for compactness normalization.
        dispersion_suppression_thresh (float): norm_disp threshold above which score is 0.
        dispersion_scale (float): control mapping sharpness of dispersion score.

    Returns:
        dict: scores for compactness, solidity, centroid_dispersion, final_score
    """
    assert np.isclose(np.sum(score_ratio), 1.0), "score_ratio must sum to 1.0"

    inst_mask = inst_mask.astype(np.uint8)
    if np.count_nonzero(inst_mask) == 0:
        return {
            'compactness': 0.0,
            'solidity': 0.0,
            'centroid_dispersion': 0.0,
            'final': 0.0
        }

    # === Area ===
    area = np.sum(inst_mask)

    # === Contour & Perimeter ===
    contours, _ = cv2.findContours(inst_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perimeter = sum(cv2.arcLength(cnt, True) for cnt in contours)

    # === Compactness Score ===
    c_min, c_max = compactness_range
    compactness = (perimeter ** 2) / area if area > 0 else c_max
    compactness_score = 1 - (np.log(compactness / c_min) / np.log(c_max / c_min))
    compactness_score = np.clip(compactness_score, 0, 1)

    # === Solidity Score ===
    # Combine all contour points for convex hull
    all_points = np.vstack(contours).squeeze()
    if all_points.ndim == 1:  # Only one point
        solidity_score = 0.0
    else:
        try:
            hull = ConvexHull(all_points)
            convex_area = hull.volume
            solidity = area / convex_area if convex_area > 0 else 0.0
            solidity_score = np.clip(solidity, 0, 1)
        except:
            solidity_score = 0.0  # Fallback if convex hull fails

    # === Centroid Dispersion Score ===
    yx = np.column_stack(np.nonzero(inst_mask))
    centroid = yx.mean(axis=0)
    dists = np.linalg.norm(yx - centroid, axis=1)
    dispersion = np.std(dists)
    norm_dispersion = dispersion / np.sqrt(area) if area > 0 else dispersion
    if norm_dispersion >= dispersion_suppression_thresh:
        centroid_dispersion_score = 0.0
    else:
        centroid_dispersion_score = 1 - (norm_dispersion / dispersion_suppression_thresh) ** dispersion_scale
        centroid_dispersion_score = np.clip(centroid_dispersion_score, 0, 1)

    # === Final Score ===
    w1, w2, w3 = score_ratio
    final_score = w1 * compactness_score + w2 * solidity_score + w3 * centroid_dispersion_score

    return {
        'compactness': compactness_score,
        'solidity': solidity_score,
        'centroid_dispersion': centroid_dispersion_score,
        'final': final_score
    }

def mask_score_fn(inst_mask,
        score_ratio=(0.5, 0.5),
        compactness_range=(4 * np.pi, 200)):
    """
    Compute shape quality scores for a binary instance mask.

    Args:
        inst_mask (ndarray): 2D bool or 0/1 array, instance mask.
        weights (tuple): Weights for (compactness, solidity, centroid_dispersion).
        compactness_range (tuple): (min, max) for compactness normalization.
        dispersion_suppression_thresh (float): norm_disp threshold above which score is 0.
        dispersion_scale (float): control mapping sharpness of dispersion score.

    Returns:
        dict: scores for compactness, solidity, centroid_dispersion, final_score
    """
    assert np.isclose(np.sum(score_ratio), 1.0), "score_ratio must sum to 1.0"

    if np.count_nonzero(inst_mask) == 0:
        return {
            'compactness': 0.0,
            'solidity': 0.0,
            'final': 0.0
        }

    inst_mask = inst_mask.astype(np.uint8)
    
    # === Area ===
    area = np.sum(inst_mask)
    
    # === Contour & Perimeter ===
    # Use RETR_EXTERNAL and CHAIN_APPROX_SIMPLE for faster contour finding
    contours, _ = cv2.findContours(inst_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Optimized perimeter calculation
    if len(contours) == 1:
        perimeter = cv2.arcLength(contours[0], True)
    else:
        perimeter = sum(cv2.arcLength(cnt, True) for cnt in contours)

    # === Compactness Score ===
    c_min, c_max = compactness_range
    if area == 0:
        compactness = c_max
    else:
        compactness = (perimeter ** 2) / area
    
    # Precompute log ratio
    log_ratio = np.log(c_max / c_min)
    compactness_score = 1 - (np.log(max(compactness, c_min) / c_min) / log_ratio)
    compactness_score = np.clip(compactness_score, 0, 1)

    # === Solidity Score ===
    if len(contours) == 0 or area == 0:
        solidity_score = 0.0
    else:
        # Combine contours more efficiently
        all_points = np.concatenate([cnt.squeeze() for cnt in contours if len(cnt) >= 2])
        
        if len(all_points) < 3:  # Need at least 3 points for convex hull
            solidity_score = 0.0
        else:
            try:
                hull = ConvexHull(all_points)
                convex_area = hull.volume
                solidity = area / convex_area if convex_area > 0 else 0.0
                solidity_score = np.clip(solidity, 0, 1)
            except:
                solidity_score = 0.0

    # === Final Score ===
    w1, w2 = score_ratio
    final_score = w1 * compactness_score + w2 * solidity_score

    return {
        'compactness': compactness_score,
        'solidity': solidity_score,
        'final': final_score
    }

def format_mask(instmask, ctype, minsize=50):
    H,W = instmask.shape
    shiftlen = 10
    masklist = []
    slices = ndimage.find_objects(instmask)
    
    for instid, slc in enumerate(slices, start=1):
        y1, y2 = max(0, slc[0].start - shiftlen), min(H, slc[0].stop + shiftlen)
        x1, x2 = max(0, slc[1].start - shiftlen), min(W, slc[1].stop + shiftlen)
        w, h = x2 - x1, y2 - y1
        if w>minsize and h>minsize:
            # annmask = instmask == instid
            # rle = mask_utils.encode(np.asfortranarray(annmask))
            # rle['counts'] = rle['counts'].decode('utf-8')
            inst_crop = (instmask[y1:y2, x1:x2] == instid)
            masklist.append({
                "id": instid,
                "image_id": -1,
                "category_id": -1,
                # "segmentation": rle,
                "bbox": [x1,y1,w,h],
                "area": w*h,
                "iscrowd": 0,
                "ctype": ctype,
                # "scores": mask_score_fn(inst_crop),
                "scores": {'final': 1.},
            })
    return masklist

def merge_multimask(masklist, iou_thresh=0.8, size_thresh=30):
    """
    对 masklist 执行基于 mask IoU 的 NMS，保留高分 mask，去除低分重叠。
    
    Args:
        masklist (list): 包含多个 dict，每个 dict 至少包含 'segmentation'（RLE） 和 'scores'。
        iou_thresh (float): IoU 阈值，大于此阈值的低分 mask 将被移除。
        
    Returns:
        List of filtered masks.
    """
    if len(masklist) == 0:
        return []
    
    small_objs,large_objs = [],[]
    for m in masklist:
        x, y, w, h = m['bbox']
        if w < size_thresh and h < size_thresh:
            small_objs.append(m)
        else:
            large_objs.append(m)

    if not large_objs:
        return small_objs

    # 构建 bboxes 和 scores
    bboxes,scores = [],[]
    for m in large_objs:
        x, y, w, h = m['bbox']
        bboxes.append([x, y, x + w, y + h])
        scores.append(m["scores"]["final"])

    bboxes = torch.tensor(bboxes, dtype=torch.float32)
    scores = torch.tensor(scores, dtype=torch.float32)

    keep_indices = nms(bboxes, scores, iou_thresh)
    filtered_large_objs = [large_objs[i] for i in keep_indices]

    return small_objs + filtered_large_objs

def infer_single_img(img_RGB, model):
    cell_config = {
        'nucleus': dict(dia=30, flowThr=0.6, cellprobThr=0.2, min_size=15),
        'cytoplasm': dict(dia=120, flowThr=0.8, cellprobThr=0.1, min_size=10*10),
        'cluster': dict(dia=240, flowThr=-1, cellprobThr=0.1, min_size=50*50),
    }
    
    multi_masks = {}
    for ctype,config in cell_config.items():
        flowThr,dia = config['flowThr'],float(config['dia'])
        cellprobThr,minSize = config['cellprobThr'], config['min_size']
        masks_pred, results, styles = model.eval([img_RGB], batch_size=64, 
            flow_threshold=flowThr, diameter=dia, compute_masks=False)
        flowi, dP, cellprob = results[0]

        if ctype == 'cytoplasm' or ctype == 'nucleus':
            cellprob,boundary_mask = flow_to_cell_prob(dP)
            cellprob[boundary_mask] = 0.
            maski = dynamics.resize_and_compute_masks(
                    dP, cellprob,
                    cellprob_threshold=cellprobThr,
                    flow_threshold=flowThr, resize=None,
                    min_size=minSize, max_size_fraction=0.9,
                    device=model.device)
        else:
            cellprob,boundary_mask = flow_to_cell_prob(dP)
            cellprob[boundary_mask] = 0.
            binary = (cellprob > cellprobThr).astype(np.uint8)
            num_labels, labels = cv2.connectedComponents(binary, connectivity=8)
            # labels = postprocess(cellprob, boundary_mask.astype(float))
            maski = labels.astype(np.int32)
            maski = utils.fill_holes_and_remove_small_masks(maski, min_size=minSize)
        
        multi_masks[ctype] = {
            'maski': maski,
            'dia': dia,
            'flowi': flowi,
            'cellprob': cellprob
        }
    
    return multi_masks

def infer_multilevel_testdemo():
    model = models.CellposeModel(gpu=True, 
                                 pretrained_model='/x22201018/.cellpose/models/cpsam',
                                 device=torch.device("cuda:0"))
    purenames = ['1657bj008_0001','1657bj008_0096']
    for purename in purenames:
        mask_savedir = f'statistic_results/cellpose_infer/hmchh_demo/{purename}'
        os.makedirs(mask_savedir, exist_ok=True, mode=0o777)

        imgpath = f'data_resource/HMCHH/JPEGImages/{purename}.png'
        img = cv2.imread(imgpath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        multi_masks = infer_single_img(img, model)

        fig, axs = plt.subplots(3, 3, figsize=(12, 12))
        axs = axs.flatten()
        idx_current = 0

        for ctype,maskitem in multi_masks.items():
            maski,dia = maskitem['maski'],maskitem['dia']
            flowi,cellprob = maskitem['flowi'],maskitem['cellprob']
            
            np.save(f'{mask_savedir}/{ctype}.npy', maski)
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
        visual_saveroot = 'statistic_results/cellpose_infer/hmchh_demo'
        os.makedirs(visual_saveroot, exist_ok=True, mode=0o777)
        plt.savefig(f'{visual_saveroot}/{purename}_cprob0.1.png')
        plt.close()

def merge_multilevel_testdemo():
    purenames = ['1657bj008_0001','1657bj008_0096']
    for purename in purenames:
        img_url = f'data_resource/HMCHH/JPEGImages/{purename}.png'
        img = cv2.imread(img_url)

        mask_savedir = f'statistic_results/cellpose_infer/hmchh_demo/{purename}'
        multi_masks = []
        for ctype in ['nucleus', 'cytoplasm', 'cluster']:
            instmask = np.load(f'{mask_savedir}/{ctype}.npy')
            mask_instlist = format_mask(instmask, ctype, minsize=0)
            multi_masks.extend(mask_instlist)

        # 查看面积 > minsize 且得分在前100的 mask
        # for ms in [30, 50]:
        #     # for primary_metric in ['compactness', 'solidity', 'centroid_dispersion', 'final']:
        #     for primary_metric in ['final']:
        #         visual_scored_masks(img, multi_masks, primary_metric, save_path=f'{mask_savedir}/ms{ms}_{primary_metric}_top100.png', topk=100, minsize=ms*ms, mode='top')
        #         visual_scored_masks(img, multi_masks, primary_metric, save_path=f'{mask_savedir}/ms{ms}_{primary_metric}_bottom100.png', topk=100, minsize=ms*ms, mode='bottom')
        
        merged_mask = merge_multimask(multi_masks, iou_thresh=0.6, size_thresh=5)
        visual_merged_masks(img, merged_mask, save_path=f'{mask_savedir}/merged_result_optim.png', size_thresh=50)

def infer_multilevel(begin, end, device):

    model = models.CellposeModel(gpu=True, 
                                 pretrained_model='/x22201018/.cellpose/models/cpsam',
                                 device=device)

    for imgitem in tqdm(total_all_imgitems[begin:end], ncols=80, desc=f'Infering {begin}~{end}'):
        purename = imgitem["file_name"].split('.')[0]
        mask_savedir = f'{root_dir}/cellpose_infer/{purename}'
        if os.path.exists(mask_savedir):
            continue
        os.makedirs(mask_savedir, exist_ok=True, mode=0o777)
        
        img_url = f'{root_dir}/JPEGImages/{purename}.png'
        img = cv2.imread(img_url)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        multi_masks = infer_single_img(img, model)

        for ctype,maskitem in multi_masks.items():
            np.save(f'{mask_savedir}/{ctype}.npy', maskitem['maski'])

def merge_multilevel():
    
    for tag in ['fold1_train', 'fold1_val']:
        jsonfile = f'{root_dir}/annofiles_roi/{tag}.json'
        coco_metric = CocoMetric(
            ann_file=jsonfile,
            metric='proposal',
            classwise=False,
            iou_thrs=[0.3],
            proposal_nums=(100, 300, 1000)
        )
        coco_metric.dataset_meta = dict(classes=['abnormal'])
        with open(jsonfile, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
    
        for imgitem in tqdm(json_data['images'], ncols=80):
            purename = imgitem["file_name"].split('.')[0]
            mask_savedir = f'{root_dir}/cellpose_infer/{purename}'

            multi_masks = []
            for ctype in ['nucleus', 'cytoplasm', 'cluster']:
                instmask = np.load(f'{mask_savedir}/{ctype}.npy')
                mask_instlist = format_mask(instmask, ctype, minsize=5)
                multi_masks.extend(mask_instlist)
            
            merged_mask = merge_multimask(multi_masks, iou_thresh=0.6, size_thresh=5)
            with open(f'{mask_savedir}/merged_result.json', 'w', encoding='utf-8') as f:
                json.dump(merged_mask, f, ensure_ascii=False)
            
            pred_bboxes,pred_scores = [],[]
            for maskitem in merged_mask:
                x, y, w, h = maskitem['bbox']
                if w<20 and h<20:
                    continue
                pred_bboxes.append([x, y, x + w, y + h])
                pred_scores.append(maskitem["scores"]["final"])

            pred_bboxes = torch.as_tensor(pred_bboxes)
            pred_scores = torch.as_tensor(pred_scores)
            pred_labels = torch.as_tensor([0] * len(pred_bboxes))

            # 计算每个框的面积，按面积从大到小排序
            areas = (pred_bboxes[:, 2] - pred_bboxes[:, 0]) * (pred_bboxes[:, 3] - pred_bboxes[:, 1])
            sorted_indices = torch.argsort(areas, descending=True)
            pred_instances = dict(
                bboxes=pred_bboxes[sorted_indices],
                scores=pred_scores[sorted_indices],
                labels=pred_labels[sorted_indices],
            )

            coco_metric.process(
            {},
            [dict(pred_instances=pred_instances, 
                img_id=imgitem['id'], ori_shape=(imgitem['width'], imgitem['height']))])

        print(f'Eval {tag}:')
        eval_results = coco_metric.evaluate(size=len(json_data['images']))
        print(eval_results)
            
def merge_multilevel_fast():
    
    for tag in ['fold1_train', 'fold1_val']:
        jsonfile = f'{root_dir}/annofiles_roi/{tag}.json'
        coco_metric = CocoMetric(
            ann_file=jsonfile,
            metric='proposal',
            classwise=False,
            # iou_thrs=[0.5],
            proposal_nums=(100, 300, 1000)
        )
        coco_metric.dataset_meta = dict(classes=['abnormal'])
        with open(jsonfile, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
    
        for imgitem in tqdm(json_data['images'], ncols=80):
            purename = imgitem["file_name"].split('.')[0]
            mask_savedir = f'{root_dir}/cellpose_infer/{purename}'
            with open(f'{mask_savedir}/merged_result.json', 'r', encoding='utf-8') as f:
                merged_mask = json.load(f)

            pred_bboxes,pred_scores = [],[]
            for maskitem in merged_mask:
                x, y, w, h = maskitem['bbox']
                if w<20 and h<20:
                    continue
                pred_bboxes.append([x, y, x + w, y + h])
                pred_scores.append(maskitem["scores"]["final"])

            pred_bboxes = torch.as_tensor(pred_bboxes)
            pred_scores = torch.as_tensor(pred_scores)
            pred_labels = torch.as_tensor([0] * len(pred_bboxes))

            # 计算每个框的面积，按面积从大到小排序
            areas = (pred_bboxes[:, 2] - pred_bboxes[:, 0]) * (pred_bboxes[:, 3] - pred_bboxes[:, 1])
            sorted_indices = torch.argsort(areas, descending=True)
            pred_instances = dict(
                bboxes=pred_bboxes[sorted_indices],
                scores=pred_scores[sorted_indices],
                labels=pred_labels[sorted_indices],
            )

            coco_metric.process(
            {},
            [dict(pred_instances=pred_instances, 
                img_id=imgitem['id'], ori_shape=(imgitem['width'], imgitem['height']))])

        print(f'Eval {tag}:')
        eval_results = coco_metric.evaluate(size=len(json_data['images']))
        print(eval_results)
 
def mergeresult2coco():
    for tag in ['fold1_train', 'fold1_val']:
        jsonfile = f'{root_dir}/annofiles_roi/{tag}.json'
        with open(jsonfile, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        predinfo = []
        for imgitem in tqdm(json_data['images'], ncols=80):
            purename = imgitem["file_name"].split('.')[0]
            mask_savedir = f'{root_dir}/cellpose_infer/{purename}'
            with open(f'{mask_savedir}/merged_result.json', 'r', encoding='utf-8') as f:
                merged_mask = json.load(f)

            filtered_bboxes = [
                maskitem['bbox'] for maskitem in merged_mask
                if maskitem['bbox'][2] >= 20 and maskitem['bbox'][3] >= 20
            ]

            pred_bboxes = torch.tensor(filtered_bboxes)
            areas = pred_bboxes[:, 2] * pred_bboxes[:, 3]
            sorted_indices = torch.argsort(areas, descending=True)

            for bbox in pred_bboxes[sorted_indices]:
                predinfo.append({
                    'image_id': imgitem['id'],
                    'category_id': 1,
                    'score': 1.,
                    'bbox': bbox.tolist()  # 转为 list 以避免 JSON 序列化错误
                })

        with open(f'data_resource/HMCHH/proposals_file/{tag}.json', 'w', encoding='utf-8') as f:
            json.dump(predinfo, f, ensure_ascii=False)

if __name__ == "__main__":
    # infer_multilevel_testdemo()
    # merge_multilevel_testdemo()

    root_dir = 'data_resource/HMCHH'
    # total_all_imgitems = []
    # for mode in ['train', 'val']:
    #     with open(f'{root_dir}/annofiles_roi/fold1_{mode}.json', 'r', encoding='utf-8') as f:
    #         json_data = json.load(f)
    #         total_all_imgitems.extend(json_data['images'])
    # infer_multilevel(0,len(total_all_imgitems),torch.device("cuda:2"))
    
    proposal_savedir = f'{root_dir}/proposals_file'
    os.makedirs(proposal_savedir, exist_ok=True, mode=0o777)
    # merge_multilevel()
    merge_multilevel_fast()
    # mergeresult2coco()

'''
fold1_train:
 
 Average Recall     (AR) @[ IoU=0.30:0.30 | area=   all | maxDets=100 ] = 0.725
 Average Recall     (AR) @[ IoU=0.30:0.30 | area=   all | maxDets=300 ] = 0.921
 Average Recall     (AR) @[ IoU=0.30:0.30 | area=   all | maxDets=1000 ] = 0.923
 Average Recall     (AR) @[ IoU=0.30:0.30 | area= small | maxDets=1000 ] = 0.143
 Average Recall     (AR) @[ IoU=0.30:0.30 | area=medium | maxDets=1000 ] = 0.849
 Average Recall     (AR) @[ IoU=0.30:0.30 | area= large | maxDets=1000 ] = 0.946

 Average Recall     (AR) @[ IoU=0.50:0.50 | area=   all | maxDets=100 ] = 0.642
 Average Recall     (AR) @[ IoU=0.50:0.50 | area=   all | maxDets=300 ] = 0.821
 Average Recall     (AR) @[ IoU=0.50:0.50 | area=   all | maxDets=1000 ] = 0.823
 Average Recall     (AR) @[ IoU=0.50:0.50 | area=medium | maxDets=1000 ] = 0.770
 Average Recall     (AR) @[ IoU=0.50:0.50 | area= large | maxDets=1000 ] = 0.840

 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.347
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.431
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.432
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.350
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.456
'''