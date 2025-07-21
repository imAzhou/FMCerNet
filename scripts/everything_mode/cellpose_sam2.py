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
from cellpose import models,utils,transforms,dynamics
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from pycocotools import mask as mask_utils
from mmdet.evaluation import CocoMetric
from collections import defaultdict
from tqdm import tqdm
from sam2.utils.amg import batch_iterator

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

def format_mask(instmask):
    H,W = instmask.shape
    masklist = []
    slices = ndimage.find_objects(instmask)
    for instid, slc in enumerate(slices, start=1):
        y1, y2 = max(0, slc[0].start), min(H, slc[0].stop)
        x1, x2 = max(0, slc[1].start), min(W, slc[1].stop)
        w, h = x2 - x1, y2 - y1
        # rle = mask_utils.encode(np.asfortranarray(instmask==instid))
        # rle['counts'] = rle['counts'].decode('utf-8')
        masklist.append({
            # "segmentation": rle,
            "bbox": [x1,y1,x2,y2],
            'cxcy':[x1 + w/2, y1 + h/2],
            'w': w,
            'h': h,
            "scores": 1.,
        })
    return masklist

def infer_single_img(img_RGB, cellpose_model, sam2_model, usesam2_sizethr):
    cell_config = {
        'nucleus': dict(dia=30, flowThr=0.6, cellprobThr=0.2, min_size=15),
        'cytoplasm': dict(dia=120, flowThr=0.8, cellprobThr=0.1, min_size=10*10),
        'cluster': dict(dia=240, flowThr=-1, cellprobThr=0.1, min_size=50*50),
    }
    points_per_batch = 128
    
    mask_instlist = []
    for ctype,config in cell_config.items():
        flowThr,dia = config['flowThr'],float(config['dia'])
        cellprobThr,minSize = config['cellprobThr'], config['min_size']
        masks_pred, results, styles = cellpose_model.eval([img_RGB], batch_size=64, 
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
                    device=cellpose_model.device)
        else:
            cellprob,boundary_mask = flow_to_cell_prob(dP)
            cellprob[boundary_mask] = 0.
            binary = (cellprob > cellprobThr).astype(np.uint8)
            num_labels, labels = cv2.connectedComponents(binary, connectivity=8)
            # labels = postprocess(cellprob, boundary_mask.astype(float))
            maski = labels.astype(np.int32)
            maski = utils.fill_holes_and_remove_small_masks(maski, min_size=minSize)
        
        mask_instlist.extend(format_mask(maski))

    small_masklist = [item for item in mask_instlist if (item['w']<usesam2_sizethr or item['h']<usesam2_sizethr)]
    kept_masks,kept_scores,kept_points,kept_bboxes = [],[],[],[]
    if len(small_masklist) > 0:
        sam2_model.set_image(img_RGB)
        mask_center = np.array([[item['cxcy']] for item in small_masklist]) # (k,1,2)
        allbatch_masks, allbatch_scores = [],[]
        for (points,) in batch_iterator(points_per_batch, mask_center):
            mask_labels = np.ones((len(points),1)) # (k,1)
            # masks: (k,3,H,W),scores: (k,3)
            masks, scores, _ = sam2_model.predict(
                point_coords=points,
                point_labels=mask_labels,
                box=None,
                multimask_output=True,
            )
            if len(masks.shape) == 3:
                masks = masks[None,:]
                scores = scores[None,:]
            allbatch_masks.extend(masks)
            allbatch_scores.extend(scores)

        for masklist, scorelist, coord in zip(allbatch_masks, allbatch_scores, mask_center):
            for mask,score in zip(masklist, scorelist):
                if  np.sum(mask) > 30*30:
                    ys, xs = np.where(mask)
                    x1,x2 = np.min(xs), np.max(xs)
                    y1,y2 = np.min(ys), np.max(ys)
                    w,h = x2-x1, y2-y1

                    if score>0.1 and (w*h) < (2048*2048)*0.3:
                        kept_masks.append(mask)
                        kept_scores.append(score)
                        kept_points.append(coord)
                        kept_bboxes.append([x1,y1,x2,y2])
    
    # 转为 tensor 用于 NMS
    total_bboxes = [item['bbox'] for item in mask_instlist if (item['w']>usesam2_sizethr and item['h']>usesam2_sizethr)]
    total_scores = [1.]*len(total_bboxes)
    total_bboxes.extend(kept_bboxes)
    total_scores.extend(kept_scores)
    bboxes = torch.tensor(total_bboxes, dtype=torch.float32)  # (N, 4)
    scores = torch.tensor(total_scores, dtype=torch.float32)  # (N,)

    # NMS
    nms_indices = nms(bboxes, scores, iou_threshold=0.5)

    # 根据 NMS 结果筛选
    # final_masks = [kept_masks[i] for i in nms_indices]
    # final_scores = [total_scores[i] for i in nms_indices]
    final_bboxes = torch.tensor([total_bboxes[i] for i in nms_indices]).tolist()
    # final_points = [kept_points[i] for i in nms_indices]

    return final_bboxes

def visual_imgmask(img_RGB, bboxeslist, savepath):
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    plt.imshow(img_RGB)
    for (x1,y1,x2,y2) in bboxeslist:
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='lime', linewidth=2, facecolor='none')
        ax.add_patch(rect)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()

def infer(GPU_idx):
    cellpose_model,sam2predictor = get_models(GPU_idx)
    for tag in ['fold1_train', 'fold1_val']:
        jsonfile = f'{root_dir}/annofiles_roi/{tag}.json'
        with open(jsonfile, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        set_split = np.array_split(range(len(json_data['images'])), 8)
        process_group = [json_data['images'][i] for i in set_split[GPU_idx]]

        predinfo = defaultdict(list)
        for imgitem in tqdm(process_group, ncols=80):
            purename = imgitem["file_name"].split('.')[0]
            imgpath = f'{root_dir}/JPEGImages/{purename}.png'
            img = cv2.imread(imgpath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            final_bboxes = infer_single_img(img, cellpose_model, sam2predictor, 50)
            predinfo[imgitem['id']] = final_bboxes
            
        with open(f'{proposal_savedir}/{tag}_{GPU_idx}.json', 'w', encoding='utf-8') as f:
            json.dump(predinfo, f, ensure_ascii=False)

def demo_test():
    cellpose_model,sam2predictor = get_models()

    purenames = ['1657bj008_0150','1662bj013_0096', '1657bj008_0066']
    for purename in purenames:
        imgpath = f'data_resource/HMCHH/JPEGImages/{purename}.png'
        img = cv2.imread(imgpath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        final_bboxes = infer_single_img(img, cellpose_model, sam2predictor, 50)
        
        visual_saveroot = 'statistic_results/cellpose_infer/hmchh_demo/cellpose_sam2'
        os.makedirs(visual_saveroot, exist_ok=True, mode=0o777)
        savepath = f'{visual_saveroot}/{purename}.png'
        visual_imgmask(img, final_bboxes, savepath)

def get_models(GPU_idx):
    device = torch.device(f"cuda:{GPU_idx}")
    cellpose_ckpt = 'checkpoints/cpsam'
    cellpose_model = models.CellposeModel(gpu=True, pretrained_model=cellpose_ckpt, device=device)
    
    sam2_ckpt = "checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2_model = build_sam2(model_cfg, sam2_ckpt, device=device)
    sam2predictor = SAM2ImagePredictor(sam2_model)

    return cellpose_model,sam2predictor

def eval_metric():
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
            gt_data = json.load(f)

        total_propsals = {}
        for i in range(8):
            with open(f'{proposal_savedir}/{tag}_{i}.json', 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            total_propsals.update(json_data)
        
        cnt = 0
        for imgitem in tqdm(gt_data['images'], ncols=80):
            # if cnt > 50:
            #     break
            # purename = imgitem["file_name"].split('.')[0]
            # imgpath = f'data_resource/HMCHH/JPEGImages/{purename}.png'
            # img = cv2.imread(imgpath)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # os.makedirs('statistic_results/HMCHH/cp_sam2', exist_ok=True, mode=0o777)
            # savepath = f'statistic_results/HMCHH/cp_sam2/{purename}.png'
            # visual_imgmask(img, total_propsals[str(imgitem['id'])], savepath)
            # cnt += 1

            filtered_bboxes = []
            for x1,y1,x2,y2 in total_propsals[str(imgitem['id'])]:
                w, h = x2-x1, y2-y1
                if w<10 and h<10:
                    continue
                if w<50 and h<50:
                    shift = 10
                elif w<100 and h<100:
                    shift = 15
                else:
                    shift = 20
                filtered_bboxes.append([x1-shift, y1-shift, x2+shift, y2+shift])

            pred_bboxes = torch.as_tensor(filtered_bboxes)
            pred_scores = torch.as_tensor([1.] * len(pred_bboxes))
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
        eval_results = coco_metric.evaluate(size=len(gt_data['images']))
        print(eval_results)


if __name__ == "__main__":
    root_dir = 'data_resource/HMCHH'
    proposal_savedir = f'{root_dir}/proposals_file/cp_sam2'
    os.makedirs(proposal_savedir, exist_ok=True, mode=0o777)

    # demo_test()
    # GPU_idx = 0
    # infer(GPU_idx)

    eval_metric()
