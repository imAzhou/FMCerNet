import json
import pickle
from pycocotools.coco import COCO
import numpy as np
from tqdm import tqdm
import cv2
import os

# 定义一个全局变量，将在 __main__ 中被赋值
# visual_missed 函数会依赖这个全局变量

def bbox_iou(box1, box2):
    """计算两个bbox的IoU，输入为 (x1,y1,x2,y2)"""
    # 交集区域
    inter_x1 = np.maximum(box1[0], box2[0])
    inter_y1 = np.maximum(box1[1], box2[1])
    inter_x2 = np.minimum(box1[2], box2[2])
    inter_y2 = np.minimum(box1[3], box2[3])

    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    # 各自面积
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = area1 + area2 - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / union_area

def analyze_unfound_gt(gt_coco_jsonfile, pkl_filepath, save_txt_path, iou_thr=0.3):
    coco = COCO(gt_coco_jsonfile)
    with open(gt_coco_jsonfile, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    with open(pkl_filepath, 'rb') as f:
        loaded_data = pickle.load(f)

    classes = [i['name'] for i in gt_data['categories']]

    # 未被找到的GT记录
    unfound_stats = {
        'small': {cls: 0 for cls in classes},
        'medium': {cls: 0 for cls in classes},
        'large': {cls: 0 for cls in classes},
    }

    # 全部GT类别统计
    total_gt_stats = {cls: 0 for cls in classes}

    for imgitem in tqdm(gt_data['images'], ncols=80, desc='Eval Metric (Analyze GT)'):
        filename = imgitem['file_name']  #格式: total_pos/JFSW_...png
        ann_ids = coco.getAnnIds(imgIds=[imgitem['id']])
        anns = coco.loadAnns(ann_ids)

        #对于每一张GT图像，找到它对应得推理图像信息
        # PKL文件中键的样子是：total_pos/JFSW_2_713_2483064819692_4.png 这样得，前面无train
        proposal_instance_data = loaded_data.get(filename, None) #从推理的PKL文件中的得到filename对应的 box,score,label信息
        
        if proposal_instance_data is None:
            # PKL 文件中没有此图片
            # 将此图像的所有 GT 视为 "未找到"
            proposal_bboxes = np.empty((0, 4), dtype=np.float32)
        else:
            # 安全地获取 bboxes (处理 AttributeError)
            bboxes_data = proposal_instance_data.bboxes
            
            if isinstance(bboxes_data, np.ndarray):
                # 已经是 NumPy 数组
                proposal_bboxes = bboxes_data.astype(np.float32)
            elif hasattr(bboxes_data, 'cpu') and hasattr(bboxes_data, 'numpy'):
                # 是 PyTorch Tensor，需要转换
                proposal_bboxes = bboxes_data.cpu().numpy().astype(np.float32)
            else:
                # 无法识别的格式，视为空
                proposal_bboxes = np.empty((0, 4), dtype=np.float32)

            # 确保 (N, 4) 格式
            if proposal_bboxes.ndim == 1:
                proposal_bboxes = proposal_bboxes.reshape(-1, 4)

        #遍历每一个GT图片得annns信息
        for ann in anns:
            gt_bbox = ann['bbox']  # [x, y, w, h]
            gt_box = np.array([
                gt_bbox[0],
                gt_bbox[1],
                gt_bbox[0] + gt_bbox[2],
                gt_bbox[1] + gt_bbox[3]
            ])
            gt_area = ann['area']
            cat_name = coco.loadCats([ann['category_id']])[0]['name']

            total_gt_stats[cat_name] += 1  # 统计总GT数量

            # IoU计算，对每一个GT框计算和其他推理框得IOU值
            if proposal_bboxes.shape[0] > 0:
                ious = [bbox_iou(gt_box, p_box) for p_box in proposal_bboxes]
                max_iou = max(ious)
            else:
                max_iou = 0.0 # 没有 proposals，最大 IoU 为 0

            if max_iou < iou_thr:  # 没找到的GT
                if gt_area < 32**2:
                    scale = 'small'
                elif gt_area < 96**2:
                    scale = 'medium'
                else:
                    scale = 'large'
                unfound_stats[scale][cat_name] += 1

    #所有GT图片已经遍历完了
    # ===== 构造输出文本 =====
    lines = []
    lines.append(f"GT JSON file: {gt_coco_jsonfile}")
    lines.append(f"PKL file: {pkl_filepath}")
    lines.append("")

    # 所有GT类别分布
    lines.append("================ 所有 GT 的类别分布 ================")
    total_gt_sum = sum(total_gt_stats.values())
    for cls, cnt in total_gt_stats.items():
        percent = 100 * cnt / total_gt_sum if total_gt_sum > 0 else 0
        lines.append(f"{cls:15s}: {cnt:6d}  ({percent:5.2f}%)")
    lines.append("")

    # 未找到的GT分布
    lines.append("================ 未被找到的 GT 统计结果 ================")
    for scale in ['small', 'medium', 'large']:
        total = sum(unfound_stats[scale].values())
        if total == 0 and scale not in ['small', 'medium', 'large']: # 确保即使为0也打印
             continue
        lines.append(f"\n[{scale.upper()}] 未被找到的总数: {total}")
        for cls, cnt in unfound_stats[scale].items():
            if cnt > 0:
                lines.append(f"  {cls:15s}: {cnt}")

    # ===== 保存到txt =====
    with open(save_txt_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

    print(f"\n统计结果已保存到: {save_txt_path}")
    return unfound_stats, total_gt_stats

def analyze_gt(gt_coco_jsonfile, save_txt_path):
    #此函数未被调用
    coco = COCO(gt_coco_jsonfile)
    
def compute_iou(box, boxes):
    """
    box: [x1, y1, x2, y2]
    boxes: (N, 4)
    return: IoU array, shape (N,)
    """
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0, xx2 - xx1)
    inter_h = np.maximum(0, yy2 - yy1)
    inter = inter_w * inter_h

    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    union = area_box + area_boxes - inter
    iou = inter / np.maximum(union, 1e-6)
    return iou

def visual_missed(gt_coco_jsonfile, pkl_filepath, visual_savedir, iou_thr):
    coco_gt = COCO(gt_coco_jsonfile)
    with open(gt_coco_jsonfile, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    with open(pkl_filepath, 'rb') as f:
        loaded_data = pickle.load(f)
    
    #遍历每一个GT图片
    for imgitem in tqdm(gt_data['images'], ncols=80, desc='Eval Metric (Visual)'):
        filename_with_prefix = imgitem['file_name'] # 格式: total_pos/JFSW_...png
        
        #就是 total_pos/JFSW_...png这样的
        filename_no_prefix = filename_with_prefix
        if filename_no_prefix.startswith('total_pos/'):
            filename_no_prefix = filename_no_prefix[len('total_pos/'):] # 移除前面得total_pos，和img_di好拼接
            
        img_path = f'{img_dir}/{filename_no_prefix}' #因为img_dir后面已经有一个total_pos了
        ann_ids = coco_gt.getAnnIds(imgIds=[imgitem['id']])
        anns = coco_gt.loadAnns(ann_ids)
        
        #对每一个GT图片得到它对应得推理信息
        # 使用完整路径 作为 PKL 键， total_pos/JFSW_...png这样的
        proposal_instance_data = loaded_data.get(filename_with_prefix, None)
        
        # ------------------- 修正 NoneType 和 AttributeError -------------------
        if proposal_instance_data is None:
            proposal_bboxes = np.empty((0, 4), dtype=np.float32)
        else:
            # 安全地获取 bboxes
            bboxes_data = proposal_instance_data.bboxes
            
            if isinstance(bboxes_data, np.ndarray):
                proposal_bboxes = bboxes_data.astype(np.float32)
            elif hasattr(bboxes_data, 'cpu') and hasattr(bboxes_data, 'numpy'):
                proposal_bboxes = bboxes_data.cpu().numpy().astype(np.float32)
            else:
                proposal_bboxes = np.empty((0, 4), dtype=np.float32) # 无法识别

            # 确保 (N, 4) 格式
            if proposal_bboxes.ndim == 1:
                proposal_bboxes = proposal_bboxes.reshape(-1, 4)

        missed_bboxes = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            ann_box = np.array([x, y, x + w, y + h])
            
            # 检查是否有 proposals 可供比较
            if proposal_bboxes.shape[0] > 0:
                ious = compute_iou(ann_box, proposal_bboxes)
                max_iou = np.max(ious)
            else:
                max_iou = 0.0 # 没有 proposals，最大 IoU 为 0
            
            if max_iou < iou_thr:
                missed_bboxes.append(ann_box)

        if len(missed_bboxes) > 0:
            img = cv2.imread(img_path)
            # 确保图像被正确读取
            if img is None:
                print(f"警告: 无法读取图像 {img_path}")
                continue
                
            # 左图：漏检 GT（红框）
            left_img = img.copy()
            for box in missed_bboxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(left_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(left_img, f"Missed GT: {len(missed_bboxes)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # 右图：所有 proposal（绿框）
            right_img = img.copy()
            if proposal_bboxes.shape[0] > 0:
                for box in proposal_bboxes:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(right_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            cv2.putText(right_img, f"Proposals: {proposal_bboxes.shape[0]}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 拼接
            combined = np.concatenate((left_img, right_img), axis=1)

            # 使用不带前缀的文件名保存，避免创建 "total_pos" 子目录
            save_path = os.path.join(visual_savedir, filename_no_prefix)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, combined)
    print(f"\n可视化结果已保存到: {visual_savedir}")


if __name__ == "__main__":
    root_dir = 'data_resource/WINDOW_SIZE_1600'
    
    #真实标签文件，训练集合加上测试集合
    gt_coco_jsonfile = f'{root_dir}/annofiles/total_cocoformat.json' 
    
    # 推理结果文件
    pkl_filepath = f'{root_dir}/train_proposal_maxDet300.pkl' 
    
    #pnmg图像文件目录
    # 声明为全局变量，供 visual_missed 使用
    global img_dir 
    img_dir = f'{root_dir}/images/total_pos'
    
    # 输出文件和参数
    proposal_savepath = f"{root_dir}/proposal_analyze_WS1600_missed.txt"
    iou_thr = 0.3 # IoU 匹配阈值
    visual_savedir = f'statistic_results/cellpose_infer/WS1600_missed'
    
    os.makedirs(visual_savedir, exist_ok=True)
    
    print(f"--- 开始分析漏检 (Missed GT) ---")
    print(f"GT JSON: {gt_coco_jsonfile}")
    print(f"PKL Proposals: {pkl_filepath}")
    print(f"Image Directory: {img_dir}")
    print("-" * 35)

    # 1. 运行统计分析 (结果保存到 .txt 文件)
    analyze_unfound_gt(gt_coco_jsonfile, pkl_filepath, proposal_savepath, iou_thr)
    
    # 2. 运行可视化分析 (结果保存到 visual_savedir 文件夹)
    visual_missed(gt_coco_jsonfile, pkl_filepath, visual_savedir, iou_thr)
    
    print(f"\n--- 漏检分析完成 ---")
