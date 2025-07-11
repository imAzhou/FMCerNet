import json
from mmdet.evaluation import CocoMetric
from tqdm import tqdm
import torch
from torchvision.ops import nms
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import random
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from mmdet.models.task_modules.prior_generators import AnchorGenerator
from mmdet.evaluation import DumpProposals

H, W = 2048, 2048
max_per_img = 1000

def augment_bboexs(d_predinfo):
    shift = 10
    too_small_thr, resetlens = 30, [150,250]   # 如果宽高均小于 too_small_thr，则以原bbox中心点将bbox宽高重设为 resetlen**2

    def generate_boxes(centerx, centery, resetlen):
        boxes = []
        wh_list = [
            (int(resetlen * 1.5), resetlen),  # 宽比高多1/2
            (resetlen, resetlen),               # 宽高相等
            (int(resetlen * 0.5), resetlen)   # 宽比高少1/2
        ]
        for w, h in wh_list:
            x1 = max(0, int(centerx - w / 2))
            y1 = max(0, int(centery - h / 2))
            x2 = min(int(centerx + w / 2), W-1)
            y2 = min(int(centery + h / 2), H-1)

            boxes.append([x1, y1, x2, y2])
        return boxes

    augmented_d_predinfo = []
    for predbbox in d_predinfo:
        x1,y1,x2,y2 = predbbox
        w,h = x2-x1, y2-y1
        if w<too_small_thr and h<too_small_thr:
            centerx,centery = x1+w/2, y1+h/2
            reset_bboxes = []
            for resetlen in resetlens:
                reset_bboxes.extend(generate_boxes(centerx,centery,resetlen))
        else:
            x1,y1,x2,y2 = x1-shift,y1-shift,x2+shift,y2+shift
            reset_bboxes = [[x1,y1,x2,y2]]
        augmented_d_predinfo.extend(reset_bboxes)
    return augmented_d_predinfo

def gene_grid_bboxes():
    anchor_generator = AnchorGenerator(
        strides=[128],               # 原图 / 特征图 = 2048 / 16 = 128
        ratios=[0.5, 1.0, 1.5],               # 宽高比
        scales=[1]                  # scale*strides 是bbox的宽高
    )
    featmap_sizes = [(16, 16)]
    anchors_list = anchor_generator.grid_priors(featmap_sizes, device='cpu')
    anchors = anchors_list[0]  # shape: [32*32*3, 4]
    anchors[:, [0, 2]] = anchors[:, [0, 2]].clamp(min=0, max=W)
    anchors[:, [1, 3]] = anchors[:, [1, 3]].clamp(min=0, max=H)
    return anchors.tolist()

def statistic_bboxes(jsonfile):
    coco = COCO(jsonfile)
    ann_ids = coco.getAnnIds()
    anns = coco.loadAnns(ann_ids)
    small, medium, large = 0, 0, 0
    for ann in anns:
        area = ann['area']
        if area < 32 ** 2:
            small += 1
        elif area < 96 ** 2:
            medium += 1
        else:
            large += 1
    print(f"Small: {small}, Medium: {medium}, Large: {large}")

def clamp_pred_fn(pred_bboxes,pred_scores,max_per_img):

    if len(pred_bboxes) > max_per_img:
        # 根据分数排序并取前 max_per_img 个
        scores_sorted, idx_sorted = torch.sort(pred_scores, descending=True)
        pred_bboxes = pred_bboxes[idx_sorted[:max_per_img]]
        pred_scores = pred_scores[idx_sorted[:max_per_img]]
    else:
        # 随机扩充 bbox
        num_to_add = max_per_img - len(pred_bboxes)
        indices = torch.randint(0, len(pred_bboxes), (num_to_add,))
        new_bboxes = pred_bboxes[indices].clone()
        
        # 随机扰动 bbox 坐标
        for i in range(num_to_add):
            x1, y1, x2, y2 = new_bboxes[i]
            dx = random.randint(50, 150)
            dy = random.randint(50, 150)
            new_x1 = max(0, min(W-1, x1.item() + random.choice([-dx, dx])))
            new_y1 = max(0, min(H-1, y1.item() + random.choice([-dy, dy])))
            new_x2 = max(0, min(W-1, x2.item() + random.choice([-dx, dx])))
            new_y2 = max(0, min(H-1, y2.item() + random.choice([-dy, dy])))
            # 保证 x2 > x1, y2 > y1，防止非法框
            new_x1, new_x2 = sorted([new_x1, new_x2])
            new_y1, new_y2 = sorted([new_y1, new_y2])
            new_bboxes[i] = torch.tensor([new_x1, new_y1, new_x2, new_y2], dtype=torch.float32)

        # 添加新框
        pred_bboxes = torch.cat([pred_bboxes, new_bboxes], dim=0)
        pred_scores = torch.cat([pred_scores, torch.ones(num_to_add)*0.5], dim=0)

    return pred_bboxes,pred_scores

def main():
    jsonfile = 'data_resource/HMCHH/annofiles_roi/fold1_train.json'
    with open(jsonfile, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    # statistic_bboxes(jsonfile)
    
    preddirs = [
        'data_resource/HMCHH/proposal_d60',
        'data_resource/HMCHH/proposal_d120'
    ]

    coco_metric = CocoMetric(
        ann_file=jsonfile,
        metric='proposal',
        classwise=False,
        iou_thrs=[0.3],
        proposal_nums=(300, 1000, 2000)
    )
    coco_metric.dataset_meta = dict(classes=['abnormal'])
    
    for imgitem in tqdm(json_data['images'], ncols=80):
        purename = imgitem["file_name"].split('.')[0]
        pred_bboxes, pred_scores = [],[]
        d_predinfo = []
        for preddir in preddirs:
            with open(f'{preddir}/{purename}.json', 'r', encoding='utf-8') as f:
                d_predinfo.extend(json.load(f))
        
        # img_predinfo.extend(d_predinfo)

        # 扩展每个bbox
        augmented_d_predinfo = augment_bboexs(d_predinfo)
        pred_bboxes.extend(augmented_d_predinfo)
        pred_scores.extend([1.0]*len(augmented_d_predinfo))
            
        # 生成规则化bboxes
        prior_bboxes = gene_grid_bboxes()
        pred_bboxes.extend(prior_bboxes)
        pred_scores.extend([0.7]*len(prior_bboxes))

        pred_bboxes = torch.as_tensor(pred_bboxes)
        pred_scores = torch.as_tensor(pred_scores)
        keep_idx = nms(pred_bboxes, pred_scores, iou_threshold=0.7)
        pred_bboxes = pred_bboxes[keep_idx]
        pred_scores = pred_scores[keep_idx]

        pred_bboxes,pred_scores = clamp_pred_fn(pred_bboxes,pred_scores,max_per_img)

        pred_instances = dict(
            bboxes=pred_bboxes,
            scores=pred_scores,
            labels=torch.as_tensor([0]*len(pred_bboxes)),
        )

        coco_metric.process(
        {},
        [dict(pred_instances=pred_instances, 
              img_id=imgitem['id'], ori_shape=(imgitem['width'], imgitem['height']))])
    
    eval_results = coco_metric.evaluate(size=len(json_data['images']))
    print(eval_results)

def visualize_fn(coco_gt, pred_info, missed_anns, image_root, num_images=5, maxDet=1000):
    """
    可视化未召回 ann 所在图像（GT vs 预测框）
    
    参数:
        coco_gt: COCO ground truth 对象
        pred_info: list[dict], 预测结果，每个包含 image_id, bbox, score, category_id
        missed_anns: list[dict], 未被召回的 anns
        image_root: 图像路径根目录
        num_images: 可视化的图像数量（默认最多显示 5 张）
    """
    import matplotlib.patches as patches
    from collections import defaultdict

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
        visual_saveroot = 'statistic_results/cellpose_infer/hmchh_missed'
        os.makedirs(visual_saveroot, exist_ok=True, mode=0o777)
        plt.savefig(f"{visual_saveroot}/{img_info['file_name']}")
        plt.close()

def missed_analyze():
    jsonfile = 'data_resource/HMCHH/annofiles_roi/fold1_train.json'
    coco_gt = COCO(jsonfile)
    with open('data_resource/HMCHH/proposals_file/fold1_train.json', 'r', encoding='utf-8') as f:
        pred_info = json.load(f)
    coco_dt = coco_gt.loadRes(pred_info)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.params.iouThrs = [0.3]  # 只评估 IoU=0.3 的情况
    maxDet = 300  # 只评估 最大候选框数量=300 的情况
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

    visualize_fn(
        coco_gt=coco_gt,
        pred_info=pred_info,
        missed_anns=missed_anns,
        image_root='data_resource/HMCHH/JPEGImages',
        num_images=50,
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

def make_infer2proposal():
    maxdet = 300
    proposal_savedir = 'data_resource/HMCHH/proposals_file'
    os.makedirs(proposal_savedir, exist_ok=True, mode=0o777)
    for tag in ['fold1_train', 'fold1_val']:
        dump_handle = DumpProposals(
            output_dir = f'{proposal_savedir}/',
            proposals_file = f'{tag}_maxdet{maxdet}.pkl',
            num_max_proposals = max_per_img
        )
        jsonfile = f'data_resource/HMCHH/annofiles_roi/{tag}.json'
        with open(jsonfile, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        for imgitem in tqdm(json_data['images'], ncols=80):
            purename = imgitem["file_name"].split('.')[0]
            mask_savedir = f'{root_dir}/cellpose_infer/{purename}'
            with open(f'{mask_savedir}/merged_result.json', 'r', encoding='utf-8') as f:
                merged_mask = json.load(f)

            pred_bboxes, pred_scores = [],[]
            for maskitem in merged_mask:
                x, y, w, h = maskitem['bbox']
                if w<20 and h<20:
                    continue
                pred_bboxes.append([x, y, x + w, y + h])
                pred_scores.append(maskitem["scores"]["final"])

            pred_bboxes = torch.as_tensor(pred_bboxes)
            pred_scores = torch.as_tensor(pred_scores)

            # 计算每个框的面积，按面积从大到小排序
            areas = (pred_bboxes[:, 2] - pred_bboxes[:, 0]) * (pred_bboxes[:, 3] - pred_bboxes[:, 1])
            sorted_indices = torch.argsort(areas, descending=True)
            sorted_indices = sorted_indices[:maxdet]
            pred_instances = dict(
                bboxes=pred_bboxes[sorted_indices],
                scores=pred_scores[sorted_indices],
            )

            dump_handle.process(None, [{
                'pred_instances': pred_instances,
                'img_path': f'data_resource/HMCHH/JPEGImages/{imgitem["file_name"]}'
            }])
        
        dump_handle.evaluate(size=len(json_data['images']))


if __name__ == "__main__":
    root_dir = 'data_resource/HMCHH'
    # main()
    # missed_analyze()
    make_infer2proposal()

    

'''
Small: 21, Medium: 2702, Large: 9455


diameter = 120
 Average Recall     (AR) @[ IoU=0.30:0.30 | area=   all | maxDets=300 ] = 0.652
 Average Recall     (AR) @[ IoU=0.30:0.30 | area=   all | maxDets=1000 ] = 0.671
 Average Recall     (AR) @[ IoU=0.30:0.30 | area=   all | maxDets=2000 ] = 0.671
 Average Recall     (AR) @[ IoU=0.30:0.30 | area=medium | maxDets=2000 ] = 0.315
 Average Recall     (AR) @[ IoU=0.30:0.30 | area= large | maxDets=2000 ] = 0.775

diameter = 120 + 扩展每个bbox
 Average Recall     (AR) @[ IoU=0.30:0.30 | area=   all | maxDets=300 ] = 0.667
 Average Recall     (AR) @[ IoU=0.30:0.30 | area=   all | maxDets=1000 ] = 0.694
 Average Recall     (AR) @[ IoU=0.30:0.30 | area=   all | maxDets=2000 ] = 0.694
 Average Recall     (AR) @[ IoU=0.30:0.30 | area=medium | maxDets=2000 ] = 0.310
 Average Recall     (AR) @[ IoU=0.30:0.30 | area= large | maxDets=2000 ] = 0.806

diameter = 120 + 扩展每个bbox + 生成规则化bboxes
 Average Recall     (AR) @[ IoU=0.30:0.30 | area=   all | maxDets=300 ] = 0.689
 Average Recall     (AR) @[ IoU=0.30:0.30 | area=   all | maxDets=1000 ] = 0.797
 Average Recall     (AR) @[ IoU=0.30:0.30 | area=   all | maxDets=2000 ] = 0.916
 Average Recall     (AR) @[ IoU=0.30:0.30 | area=medium | maxDets=2000 ] = 0.724
 Average Recall     (AR) @[ IoU=0.30:0.30 | area= large | maxDets=2000 ] = 0.973

 Average Recall     (AR) @[ IoU=0.50:0.50 | area=   all | maxDets=300 ] = 0.603
 Average Recall     (AR) @[ IoU=0.50:0.50 | area=   all | maxDets=1000 ] = 0.682
 Average Recall     (AR) @[ IoU=0.50:0.50 | area=   all | maxDets=2000 ] = 0.758
 Average Recall     (AR) @[ IoU=0.50:0.50 | area=medium | maxDets=2000 ] = 0.377
 Average Recall     (AR) @[ IoU=0.50:0.50 | area= large | maxDets=2000 ] = 0.868

diameter = 90 + 扩展每个bbox + 生成规则化bboxes
 Average Recall     (AR) @[ IoU=0.50:0.50 | area=   all | maxDets=300 ] = 0.586
 Average Recall     (AR) @[ IoU=0.50:0.50 | area=   all | maxDets=1000 ] = 0.672
 Average Recall     (AR) @[ IoU=0.50:0.50 | area=   all | maxDets=2000 ] = 0.756
 Average Recall     (AR) @[ IoU=0.50:0.50 | area=medium | maxDets=2000 ] = 0.441
 Average Recall     (AR) @[ IoU=0.50:0.50 | area= large | maxDets=2000 ] = 0.848
'''