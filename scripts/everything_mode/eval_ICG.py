import json
import warnings
import os

from tqdm import tqdm
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
warnings.filterwarnings("ignore", category=UserWarning)
import torch
from torchvision.ops import nms,box_iou
from typing import List, Tuple, Dict, Any

# ====================== cellpose infer params ======================
cervical_cell_config = {
    'nucleus': dict(dia=15, flowThr=0.6, cellprobThr=0.1, min_size=15),
    'cytoplasm': dict(dia=120, flowThr=0.8, cellprobThr=0.1, min_size=10*10),
    'cluster': dict(dia=240, flowThr=-1, cellprobThr=0.1, min_size=10*10),
}
blood_cell_config = {
    'nucleus': dict(dia=15, flowThr=0.6, cellprobThr=0.1, min_size=15),
    'cytoplasm': dict(dia=100, flowThr=0.8, cellprobThr=0.1, min_size=10*10),
    'cluster': dict(dia=150, flowThr=-1, cellprobThr=0.1, min_size=10*10),
}

dataset_config = {
    'CDetector': {
        'dataroot_dir': 'data_resource/ComparisonDetectorDataset',
        'infer_imgdir': 'train',
        'metric_json': 'train_filter_error.json',
        'cell_config': cervical_cell_config,
        'infer_result_dir': 'AutoMPG_row3'
    },
    'HMCHH': {
        'dataroot_dir': 'data_resource/HMCHH',
        'infer_imgdir': 'JPEGImages',
        'metric_json': 'annofiles_roi/fold1_train.json',
        'cell_config': cervical_cell_config,
        'infer_result_dir': 'cellpose_all_noNMS'
    },
    'HMCHHAUG': {
        'dataroot_dir': 'data_resource/HMCHH',
        'infer_imgdir': 'JPEGImages_with_augmented',
        'metric_json': 'annofiles_roi/fold1_train_with_aug.json',
        'cell_config': cervical_cell_config,
        'infer_result_dir': 'cellpose_noNMS_aug'
    },
    'CRIC': {
        'dataroot_dir': 'data_resource/CRIC',
        'infer_imgdir': 'images',
        'metric_json': 'annofiles/abnormal/fold4_train.json',
        'cell_config': cervical_cell_config
    },
    'BCCD':{
        'dataroot_dir': 'data_resource/BCCD',
        'infer_imgdir': 'train',
        'metric_json': 'annofiles/train_annotations.coco.json',
        'cell_config': blood_cell_config
    },
    'WS1600': {
        'dataroot_dir': 'data_resource/WINDOW_SIZE_1600',
        'infer_imgdir': 'images/total_pos',
        # 'metric_json': 'annofiles/puretrain_cocoformat.json'
        'metric_json': 'annofiles/total_cocoformat.json',
        'cell_config': cervical_cell_config
    },
}

DATASET_TAG = 'CDetector'

dataroot_dir = dataset_config[DATASET_TAG]['dataroot_dir']
infer_imgdir = dataset_config[DATASET_TAG]['infer_imgdir']
metric_json = dataset_config[DATASET_TAG]['metric_json']
cell_config = dataset_config[DATASET_TAG]['cell_config']
infer_result_dir = dataset_config[DATASET_TAG]['infer_result_dir']

infer_imgdirs = [
    f'{dataroot_dir}/{infer_imgdir}',
]
infer_savedir = f'{dataroot_dir}/{infer_result_dir}'
os.makedirs(infer_savedir, exist_ok=True, mode=0o777)

# =================== infer results metric params ===================
metric_jsons = [
    f'{dataroot_dir}/{metric_json}',
]
# ================== infer results format to pkl params ==================
KEEPNUM = 1000
ICG_LOW_THR = 0.2 # ICG 匹配的低 IoU 阈值
ICG_HIGH_THR = 0.5 # ICG 匹配的高 IoU 阈值
ICG_TOLERANCE = 5.0 # ICG 包含匹配的容忍度
proposal_pkl_cfg = {
    'source_cocojson': metric_jsons[0], #拼接成得路径应为data_resource/WINDOW_SIZE_1600/annofiles/total_cocoformat.json
    'output_dir': dataroot_dir,
    'pkl_filename': f'proposal_maxDet{KEEPNUM}.pkl',
    'img_dir': f'{dataroot_dir}/{infer_imgdir}',
}


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
    proposal_nums: Tuple[int, int, int]
) -> Dict[str, Any]:
    """
    计算 ICG 匹配规则下的 Average Recall (AR)。
    直接基于 ICG 匹配矩阵计算召回，不涉及多 IoU 阈值平均。
    """
    num_imgs = len(all_proposals)
    
    # 存储每个 proposal_num 下所有图像的 recall 列表
    all_recalls = {num: [] for num in proposal_nums}
    
    # 遍历每张图像
    for i in range(num_imgs):
        gt_i = all_gt_bboxes[i]
        is_matched_matrix_i = all_is_matched_ICG[i] # (M_i, K_i)

        if gt_i.size(0) == 0:
            continue

        # 遍历每个 proposal 数量限制 (e.g., 100, 300, 1000)
        for p_num in proposal_nums:
            # 截断匹配矩阵 (p_num, K_i)
            # 前提：输入进来的 is_matched_matrix_i 对应的 proposal 已经是按分数排序过的
            # 我们只看前 p_num 个 proposal 是否匹配到了 GT
            is_matched_matrix_i_trunc = is_matched_matrix_i[:p_num] 

            # GT 是否被召回的 mask (K_i,)
            # 只要该 GT 被前 p_num 个 proposal 中的任意一个匹配到，即视为召回
            gt_is_recalled = is_matched_matrix_i_trunc.any(dim=0)
            
            # 计算单张图的召回率
            recall = gt_is_recalled.sum().float() / gt_i.size(0)
            all_recalls[p_num].append(recall.item())

    # 汇总结果：计算所有有效图像的平均 Recall
    final_ar_results = {}
    for p_num in proposal_nums:
        recalls_list = all_recalls[p_num]
        if not recalls_list:
            final_ar_results[f'AR@{p_num}'] = 0.0
        else:
            final_ar_results[f'AR@{p_num}'] = sum(recalls_list) / len(recalls_list)
            
    return final_ar_results


def eval_metric_ICG(
    proposal_nums: Tuple[int, int, int] = (100, 300, 1000)
):
    """
    使用 ICG 包含匹配规则评估候选框的召回率 (AR)。
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

