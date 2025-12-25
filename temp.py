# import os
# from cerwsi.utils import KFBSlide,kfbslide_get_associated_image_names,kfbslide_read_associated_image
# import openslide

# svs_path = '/nfs-medical3/data/浙一胃HE与Fish数据/胃癌FISH片/2025126768-FISH-HE-KL.svs'
# kfb_path = '/medical-data/data/cervix/JFSW_1109/HSIL/C202028855.kfb'
# kfbf_path = '/nfs-medical3/data/浙一胃HE与Fish数据/胃癌FISH片/2025126768-FISH-KL.kfbf'

# # source_path = kfbf_path
# # slide = KFBSlide(source_path)
# # swidth, sheight = slide.level_dimensions[0]
# # associated_images = kfbslide_get_associated_image_names(slide._osr)
# # if 'label' not in associated_images:
# #     print(f'{source_path} haven\'t label!')
# # else:
# #     filename = os.path.splitext(os.path.basename(source_path))[0]
# #     image = kfbslide_read_associated_image(slide._osr, 'label')
# #     output_path = f"{filename}.png"
# #     image.save(output_path, "PNG")

# slide = openslide.OpenSlide(svs_path)
# print(slide.associated_images.keys())


import torch

data = torch.load('data_resource/0630/WINDOW_SIZE_1200/slide_feat_detector/JFSW_1_2.pt')
print()
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pycocotools import mask as mask_utils

# --- 假设的辅助函数和常量（请根据您的环境确保这些函数可用） ---

# 假设的 IoU 计算函数 (使用 torchvision 作为示例实现)
try:
    from torchvision.ops import box_iou as box_iou
    from torchvision.ops import nms as nms
except ImportError:
    # 如果 torchvision 不可用，需要自行提供 box_iou 和 nms 实现
    print("Warning: torchvision not found. Using dummy functions for box_iou and nms.")
    def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        # Placeholder: Must be replaced with actual IoU calculation
        return torch.zeros(boxes1.size(0), boxes2.size(0))
    def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
        # Placeholder: Must be replaced with actual NMS
        return torch.arange(boxes.size(0))

# 假设的常量
KEEPNUM = 1000 # 保证每张图的 proposal 数量至少达到 1000
ICG_LOW_THR = 0.2 # ICG 匹配的低 IoU 阈值
ICG_HIGH_THR = 0.5 # ICG 匹配的高 IoU 阈值
ICG_TOLERANCE = 5.0 # ICG 包含匹配的容忍度

# -----------------------------------------------------------------

def match_proposal_ICG_for_eval(
    proposals: torch.Tensor, 
    gt_bboxes: torch.Tensor, 
    low_thr: float, 
    high_thr: float, 
    tolerance: float = 5.0
) -> torch.Tensor:
    """
    使用 ICG 包含规则进行匹配，判断每个 proposal 是否为正样本。
    返回一个 (M, K) 的矩阵，表示 proposal [i] 是否成功匹配到 gt [j]。
    
    Args:
        proposals: Tensor (M, 4) in (x1, y1, x2, y2)
        gt_bboxes: Tensor (K, 4) in (x1, y1, x2, y2)

    Returns:
        is_matched_matrix: Tensor (M, K) bool. 匹配成功为 True。
    """
    M, K = proposals.size(0), gt_bboxes.size(0)
    is_matched_matrix = torch.zeros((M, K), dtype=torch.bool, device=proposals.device)

    if K == 0 or M == 0:
        return is_matched_matrix

    # 1. 计算 IoU
    ious = box_iou(proposals, gt_bboxes)  # (M, K)

    # 2. IoU 正样本匹配：只要 IoU > high_thr 就匹配成功
    pos_mask_iou = ious > high_thr
    is_matched_matrix[pos_mask_iou] = True

    # 3. ICG 包含匹配 (只对处于模糊区域的 proposal 进行)
    # 模糊区域定义：low_thr < IoU <= high_thr
    remain_mask_2d = (ious > low_thr) & (ious <= high_thr)
    
    # 获取所有处于模糊区域的 proposal 索引 (一维 M)
    remain_M_indices = torch.nonzero(remain_mask_2d.any(dim=1)).squeeze(1)

    if remain_M_indices.numel() > 0:
        remain_proposals = proposals[remain_M_indices]
        tol = tolerance
        px1, py1, px2, py2 = remain_proposals.unbind(dim=1)
        gx1, gy1, gx2, gy2 = gt_bboxes.unbind(dim=1)

        # contain_mask (M_remain, K): 检查每个 remain_proposal 是否被某个 GT 包含
        contain_mask = (
            (px1.unsqueeze(1) >= (gx1.unsqueeze(0) - tol)) &
            (py1.unsqueeze(1) >= (gy1.unsqueeze(0) - tol)) &
            (px2.unsqueeze(1) <= (gx2.unsqueeze(0) + tol)) &
            (py2.unsqueeze(1) <= (gy2.unsqueeze(0) + tol))
        )  # (M_remain, K)

        # 结合模糊区域和包含关系进行匹配
        for i in range(remain_M_indices.size(0)):
            m_idx = remain_M_indices[i] # proposal 的全局索引
            
            # 找到当前 proposal 位于模糊区域且被包含的所有 GT (K 维度)
            matched_gts = torch.nonzero(contain_mask[i] & remain_mask_2d[m_idx]).squeeze(1)
            
            if matched_gts.numel() > 0:
                # 按照原始 ICG 逻辑，找到面积最小的父 GT 进行匹配
                gt_areas = (gt_bboxes[matched_gts][:, 2:] - gt_bboxes[matched_gts][:, :2]).prod(dim=1)
                min_area_idx = matched_gts[gt_areas.argmin()]
                
                # 标记该 proposal 与面积最小的父 GT 匹配成功
                is_matched_matrix[m_idx, min_area_idx] = True

    return is_matched_matrix

def calculate_recall_ICG(
    all_proposals: List[torch.Tensor], 
    all_gt_bboxes: List[torch.Tensor],
    all_is_matched_ICG: List[torch.Tensor], # (M_i, K_i) 矩阵列表
    iou_thrs: List[float],
    proposal_nums: Tuple[int, int, int]
) -> Dict[str, Any]:
    """
    计算 ICG 匹配规则下的 COCO 风格 Average Recall (AR)。
    """
    num_imgs = len(all_proposals)
    
    # 1. 收集 GT 数量
    num_gts_per_img = [g.size(0) for g in all_gt_bboxes]
    
    # 2. 存储每个 IoU 阈值和 proposal 数量下的召回率
    all_recalls = {num: {thr: [] for thr in iou_thrs} for num in proposal_nums}
    
    # 3. 遍历每张图像
    for i in range(num_imgs):
        proposals_i = all_proposals[i]
        gt_i = all_gt_bboxes[i]
        is_matched_matrix_i = all_is_matched_ICG[i] # (M_i, K_i)

        if gt_i.size(0) == 0:
            continue

        # 4. 遍历每个 proposal 数量限制
        for p_num in proposal_nums:
            # 截断 proposal
            proposals_i_trunc = proposals_i[:p_num]
            is_matched_matrix_i_trunc = is_matched_matrix_i[:p_num] # (p_num, K_i)

            # 5. 遍历每个 IoU 阈值 (COCO AR 是针对不同 IoU 阈值的平均)
            # 注意：这里的 IoU 阈值 iou_thr 仅用于召回率的定义，而不是 ICG 匹配本身。
            # COCO 的召回率计算是：在一个 proposal 集合下，GT 成功匹配的比例。
            # 这里的 ICG 匹配结果 is_matched_matrix_i_trunc 已经包含了所有 IoU 阈值的逻辑。
            
            # 在 ICG 评估中，我们**只使用** ICG 匹配结果来判断 GT 是否被召回。
            # 如果 proposal [i] 通过 ICG 匹配到了 GT [j]，则 GT [j] 被召回。
            # 如果 proposal [i] 通过 ICG 匹配到了 GT [j]，那么对于任何 IoU 阈值，
            # 只要这个 IoU 阈值用于 AR 评估，这个 GT 就应该被视为召回。
            
            # GT 是否被召回的 mask (K_i,)
            gt_is_recalled = is_matched_matrix_i_trunc.any(dim=0)
            
            # 计算召回率
            recall = gt_is_recalled.sum().float() / gt_i.size(0)
            
            # 将该召回率计入所有评估 IoU 阈值下
            for thr in iou_thrs:
                 all_recalls[p_num][thr].append(recall.item())

    # 6. 汇总结果
    final_ar_results = {}
    
    # 计算每个 proposal_num 下的 AR (对所有 IoU 阈值和所有图像平均)
    for p_num in proposal_nums:
        recalls_list = []
        for thr in iou_thrs:
            recalls_list.extend(all_recalls[p_num][thr])
        
        if not recalls_list:
            final_ar_results[f'AR@{p_num}'] = 0.0
        else:
            final_ar_results[f'AR@{p_num}'] = sum(recalls_list) / len(recalls_list)
            
    # 计算所有 IoU 阈值和 proposal_nums 的总平均 AR (COCO 风格)
    all_recalls_flattened = []
    for p_num in proposal_nums:
        for thr in iou_thrs:
            all_recalls_flattened.extend(all_recalls[p_num][thr])
            
    final_ar_results['AR@all'] = sum(all_recalls_flattened) / len(all_recalls_flattened) if all_recalls_flattened else 0.0
    
    return final_ar_results


def eval_metric_ICG(
    metric_jsons: List[str],
    infer_savedir: str,
    iou_thrs: List[float],
    proposal_nums: Tuple[int, int, int] = (100, 300, 1000)
):
    """
    使用 ICG 包含匹配规则评估候选框的召回率 (AR)，并自实现 COCO 风格 AR 计算。
    """
    
    # 存储所有图像的 ICG 匹配数据
    all_proposals: List[torch.Tensor] = []      # 经过 NMS 和排序后的 proposal 列表
    all_gt_bboxes: List[torch.Tensor] = []      # 所有的 GT bboxes 列表
    all_is_matched_ICG: List[torch.Tensor] = [] # proposal 和 GT 之间的 ICG 匹配矩阵列表 (M_i, K_i)

    print(f"--- 🚀 ICG Proposal Metric Evaluation ---")
    print(f"ICG LOW_THR={ICG_LOW_THR}, HIGH_THR={ICG_HIGH_THR}, TOLERANCE={ICG_TOLERANCE}")
    
    for jsonfile in metric_jsons:
        with open(jsonfile, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
        
        gt_annos = {anno['image_id']: [] for anno in gt_data['annotations']}
        for anno in gt_data['annotations']:
            bbox = anno['bbox']
            # COCO format: [x, y, w, h] -> [x1, y1, x2, y2]
            gt_annos[anno['image_id']].append([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])

        for imgitem in tqdm(gt_data['images'], ncols=80, desc=f'Processing {os.path.basename(jsonfile)}'):
            img_id = imgitem['id']
            filename = imgitem['file_name']
            purename = os.path.splitext(os.path.basename(filename))[0]

            # 1. 加载预测框
            proposal_file = f'{infer_savedir}/{purename}.json'
            if not os.path.exists(proposal_file):
                # print(f"Warning: Proposal file not found for {purename}. Skipping.")
                final_bboxes_tensor = torch.empty((0, 4), dtype=torch.float32)
            else:
                with open(proposal_file, 'r', encoding='utf-8') as f:
                    proposal_bboxes_raw = json.load(f)
                
                # 2. NMS & 排序/截断 (与 eval_metric 保持一致)
                bboxes = torch.tensor(proposal_bboxes_raw, dtype=torch.float32)
                if bboxes.numel() == 0:
                     final_bboxes_tensor = torch.empty((0, 4), dtype=torch.float32)
                else:
                    widths = bboxes[:, 2] - bboxes[:, 0]
                    heights = bboxes[:, 3] - bboxes[:, 1]
                    scores = widths * heights
                    nms_indices = nms(bboxes, scores, iou_threshold=0.3)
                    final_bboxes = bboxes[nms_indices].tolist()

                    final_bboxes_sorted = sorted(
                        final_bboxes,
                        key=lambda box: (box[2] - box[0]) * (box[3] - box[1]),
                        reverse=True
                    )
                    final_bboxes_tensor = torch.as_tensor(final_bboxes_sorted[:KEEPNUM], dtype=torch.float32)

            # 3. GT Bboxes
            gt_boxes_list = gt_annos.get(img_id, [])
            gt_bboxes_tensor = torch.as_tensor(gt_boxes_list, dtype=torch.float32)
            
            # 4. ICG 匹配
            is_matched_matrix_icg = match_proposal_ICG_for_eval(
                final_bboxes_tensor, 
                gt_bboxes_tensor, 
                low_thr=ICG_LOW_THR, 
                high_thr=ICG_HIGH_THR, 
                tolerance=ICG_TOLERANCE
            )
            
            # 5. 收集结果
            all_proposals.append(final_bboxes_tensor)
            all_gt_bboxes.append(gt_bboxes_tensor)
            all_is_matched_ICG.append(is_matched_matrix_icg)
        
        # 6. 计算最终召回率 (AR)
        ar_results = calculate_recall_ICG(
            all_proposals, 
            all_gt_bboxes, 
            all_is_matched_ICG,
            iou_thrs, 
            proposal_nums
        )
        
        # 7. 格式化输出
        print(f'\nEval ICG AR in {jsonfile}:')
        print("-" * 30)
        for key in ar_results:
            print(f"{key:<8}: {ar_results[key]:.6f}")
        print("-" * 30)

    print("\nEvaluation finished.")


    

if __name__ == "__main__":
    eval_metric_ICG()
