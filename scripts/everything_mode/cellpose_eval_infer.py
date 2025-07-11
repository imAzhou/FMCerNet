import json
from mmdet.evaluation import CocoMetric
from tqdm import tqdm
import torch
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from mmdet.models.task_modules.prior_generators import AnchorGenerator
from mmdet.evaluation import DumpProposals
import random
random.seed(666)

def generate_boxes(centerx, centery, resetlen):
    boxes = []
    wh_list = [
        (int(resetlen * 1.5), resetlen),  # 宽比高多1/2
        (resetlen, resetlen),               # 宽高相等
        (int(resetlen * 0.5), resetlen)   # 宽比高少1/2
    ]
    for w, h in wh_list:
        x1 = int(centerx - w / 2)
        y1 = int(centery - h / 2)
        x2 = int(centerx + w / 2)
        y2 = int(centery + h / 2)
        boxes.append([x1, y1, x2, y2])
    return boxes

def eval_infer():
    jsonfile = 'data_resource/HMCHH/annofiles_roi/fold1_train.json'
    with open(jsonfile, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    # coco = COCO(jsonfile)
    # ann_ids = coco.getAnnIds()
    # anns = coco.loadAnns(ann_ids)
    # small, medium, large = 0, 0, 0
    # for ann in anns:
    #     area = ann['area']
    #     if area < 32 ** 2:
    #         small += 1
    #     elif area < 96 ** 2:
    #         medium += 1
    #     else:
    #         large += 1
    # print(f"Small: {small}, Medium: {medium}, Large: {large}")

    preddir = 'data_resource/HMCHH/proposal_d90'
    
    coco_metric = CocoMetric(
        ann_file=jsonfile,
        metric='proposal',
        classwise=False,
        iou_thrs=[0.5],
        proposal_nums=(300, 1000, 2000,)
        # metric_items = ['mAP', 'mAP_m', 'mAP_l', 
        #                 'AR@1000', 'AR_m@1000', 'AR_l@1000'],
        # format_only=True,
        # outfile_prefix='data_resource/HMCHH/proposal_d120'
    )
    coco_metric.dataset_meta = dict(classes=['abnormal'])
    
    shift = 10
    too_small_thr, resetlen = 50, 120   # 如果宽高均小于 too_small_thr，则以原bbox中心点将bbox宽高重设为 resetlen**2
    anchor_generator = AnchorGenerator(
        strides=[64],               # 原图 / 特征图 = 2048 / 32 = 64
        ratios=[0.5, 1.0, 1.5],               # 宽高比
        scales=[1]                  # scale*strides 是bbox的宽高
    )
    featmap_sizes = [(32, 32)]
    W,H = 2048, 2048

    bbox_cnts = []
    for imgitem in tqdm(json_data['images'], ncols=80):
        purename = imgitem["file_name"].split('.')[0]
        img_predinfo = []

        with open(f'{preddir}/{purename}.json', 'r', encoding='utf-8') as f:
            d_predinfo = json.load(f)
            # img_predinfo.extend(d_predinfo)

        # 扩展每个bbox
        for predbbox in d_predinfo:
            x1,y1,x2,y2 = predbbox
            w,h = x2-x1, y2-y1
            if w<too_small_thr and h<too_small_thr:
                centerx,centery = x1+w/2, y1+h/2
                reset_bboxes = [[x1,y1,x2,y2]]
                reset_bboxes.extend(generate_boxes(centerx,centery,resetlen))
            else:
                x1,y1,x2,y2 = x1-shift,y1-shift,x2+shift,y2+shift
                reset_bboxes = [[x1,y1,x2,y2]]
            img_predinfo.extend(reset_bboxes)
            
        # 生成规则化bboxes
        anchors_list = anchor_generator.grid_priors(featmap_sizes, device='cpu')
        anchors = anchors_list[0]  # shape: [32*32*3, 4]
        anchors[:, [0, 2]] = anchors[:, [0, 2]].clamp(min=0, max=W)
        anchors[:, [1, 3]] = anchors[:, [1, 3]].clamp(min=0, max=H)
        prior_bboxes = anchors.tolist()
        random.shuffle(prior_bboxes)
        img_predinfo.extend(prior_bboxes)

        img_predinfo = img_predinfo[:2000]

        pred_instances = dict(
            bboxes=torch.as_tensor(img_predinfo),
            scores=torch.as_tensor([1.]*len(img_predinfo)),
            labels=torch.as_tensor([0]*len(img_predinfo)),
        )
        bbox_cnts.append(len(img_predinfo))
        coco_metric.process(
        {},
        [dict(pred_instances=pred_instances, 
              img_id=imgitem['id'], ori_shape=(imgitem['width'], imgitem['height']))])
    
    eval_results = coco_metric.evaluate(size=len(json_data['images']))
    print(eval_results)

    print(f"min cnt: {min(bbox_cnts)}, max cnt: {max(bbox_cnts)}, avg cnt: {np.mean(bbox_cnts)}.")

def visualize_fn(coco_gt, pred_info, missed_anns, image_root, num_images=5):
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

        # 画预测框（蓝色）
        for pred in imgid_to_preds[img_id]:
            x, y, w, h = pred['bbox']
            axs[1].add_patch(patches.Rectangle((x, y), w, h, edgecolor='blue', facecolor='none', linewidth=1))
        
        for ax in axs:
            ax.axis('off')
        plt.tight_layout()
        visual_saveroot = 'statistic_results/cellpose_infer/hmchh'
        os.makedirs(visual_saveroot, exist_ok=True, mode=0o777)
        plt.savefig(f"{visual_saveroot}/{img_info['file_name']}")
        plt.close()

def missed_analyze():
    jsonfile = 'data_resource/HMCHH/annofiles_roi/fold1_train.json'
    coco_gt = COCO(jsonfile)
    with open('data_resource/HMCHH/proposal_d120.bbox.json', 'r', encoding='utf-8') as f:
        pred_info = json.load(f)
    coco_dt = coco_gt.loadRes(pred_info)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.params.iouThrs = [0.3]  # 只评估 IoU=0.3 的情况
    coco_eval.evaluate()
    coco_eval.accumulate()

    # 4. 提取未被召回的 ann（false negatives）
    gt_ids = coco_eval.evalImgs[0]['gtIds'] if coco_eval.evalImgs else []  # 所有 GT ann 的 id 列表
    all_gt_ids = []

    # 收集所有图片中未匹配（未召回）的 ann_id
    for evalImg in coco_eval.evalImgs:
        if evalImg is None:
            continue
        gt_ids = evalImg['gtIds']
        gt_match = evalImg['gtMatches'][0]  # IoU=0.3 的匹配状态
        for ann_id, match in zip(gt_ids, gt_match):
            if match == 0:  # 没有匹配
                all_gt_ids.append(ann_id)

    missed_anns = coco_gt.loadAnns(all_gt_ids)
    visualize_fn(
        coco_gt=coco_gt,
        pred_info=pred_info,
        missed_anns=missed_anns,
        image_root='data_resource/HMCHH/JPEGImages',
        num_images=50
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
    preddir = 'data_resource/HMCHH/proposal_d120'
    shift = 10
    too_small_thr, resetlen = 30, 100   # 如果宽高均小于 too_small_thr，则以原bbox中心点将bbox宽高重设为 resetlen**2
    anchor_generator = AnchorGenerator(
        strides=[64],               # 原图 / 特征图 = 2048 / 32 = 64
        ratios=[1.0],               # 宽高比
        scales=[1,3]                  # scale*strides 是bbox的宽高
    )
    featmap_sizes = [(32, 32)]
    W,H = 2048, 2048
    num_max_proposals = 2000

    proposal_savedir = 'data_resource/HMCHH/proposals_file'
    os.makedirs(proposal_savedir, exist_ok=True, mode=0o777)
    for tag in ['fold1_val']:
        dump_handle = DumpProposals(
            output_dir = f'{proposal_savedir}/',
            proposals_file = f'{tag}.pkl',
            num_max_proposals = num_max_proposals
        )
        jsonfile = f'data_resource/HMCHH/annofiles_roi/{tag}.json'
        with open(jsonfile, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        for imgitem in tqdm(json_data['images'], ncols=80):
            purename = imgitem["file_name"].split('.')[0]
            img_predinfo = []
            with open(f'{preddir}/{purename}.json', 'r', encoding='utf-8') as f:
                d_predinfo = json.load(f)
            # 扩展每个bbox
            for predbbox in d_predinfo:
                x1,y1,x2,y2 = predbbox
                w,h = x2-x1, y2-y1
                if w<too_small_thr and h<too_small_thr:
                    centerx,centery = x1+w/2, y1+h/2
                    reset_bboxes = generate_boxes(centerx,centery,resetlen)
                    reset_bboxes.append([x1,y1,x2,y2])
                else:
                    x1,y1,x2,y2 = x1-shift,y1-shift,x2+shift,y2+shift
                    reset_bboxes = [[x1,y1,x2,y2]]
                img_predinfo.extend(reset_bboxes)
                
            # 生成规则化bboxes
            anchors_list = anchor_generator.grid_priors(featmap_sizes, device='cpu')
            anchors = anchors_list[0]  # shape: [32*32*3, 4]
            anchors[:, [0, 2]] = anchors[:, [0, 2]].clamp(min=0, max=W)
            anchors[:, [1, 3]] = anchors[:, [1, 3]].clamp(min=0, max=H)
            prior_bboxes = anchors.tolist()
            img_predinfo.extend(prior_bboxes)

            img_predinfo = img_predinfo[:num_max_proposals]

            dump_handle.process(None, [{
                'pred_instances': {
                    'scores': torch.as_tensor([1.]*len(img_predinfo)),
                    'bboxes': torch.as_tensor(img_predinfo)
                },
                'img_path': f'data_resource/HMCHH/JPEGImages/{imgitem["file_name"]}'
            }])
        
        dump_handle.evaluate(size=len(json_data['images']))


if __name__ == "__main__":
    eval_infer()
    # missed_analyze()
    # make_infer2proposal()

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