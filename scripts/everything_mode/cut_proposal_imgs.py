import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from collections import defaultdict
import cv2
from pycocotools.coco import COCO
import torch
from mmengine.structures import InstanceData
from mmdet.models.task_modules import MaxIoUAssigner
from tqdm import tqdm
import random
random.seed(42)

def visualize_positive_matches(img_info, anns, pos_bboxes):
    """
    可视化正样本匹配结果。

    Args:
        img_info (dict): COCO格式图像信息，包含 'file_name', 'width', 'height' 等。
        anns (list): COCO中该图像的所有标注信息，每项为一个字典。
        pos_bboxes (list of dict): 每个正样本为一个字典，包含：
            - 'bbox': [x, y, w, h]
            - 'iou': float，匹配的最大IoU
    """
    # 读取原图
    img_path = os.path.join(image_root, img_info['file_name'])
    image = Image.open(img_path)

    fig, axs = plt.subplots(1, 2, figsize=(16, 8))

    # 第一列：原图 + GT bbox
    axs[0].imshow(image)
    axs[0].set_title("GT BBoxes")
    axs[0].axis('off')
    for ann in anns:
        x, y, w, h = ann['bbox']
        rect = patches.Rectangle((x, y), w, h,
                                 linewidth=2, edgecolor='green', facecolor='none')
        axs[0].add_patch(rect)

    # 第二列：原图 + 正样本bboxes
    axs[1].imshow(image)
    axs[1].set_title("Matched Positive BBoxes")
    axs[1].axis('off')
    for item in pos_bboxes:
        x, y, w, h = item['bbox']
        iou = item['iou']
        rect = patches.Rectangle((x, y), w, h,
                                 linewidth=2, edgecolor='red', facecolor='none')
        axs[1].add_patch(rect)
        axs[1].text(x, y - 2, f"IoU: {iou:.2f}", color='red', fontsize=10, verticalalignment='bottom')

    plt.tight_layout()
    save_path = f'{visual_savedir}/{img_info["file_name"]}'
    plt.savefig(save_path)
    plt.close()

def crop_save_matches(img_info, pos_bboxes, neg_bboxes):
    """
    存储正样本匹配结果。

    Args:
        img_info (dict): COCO格式图像信息，包含 'file_name', 'width', 'height' 等。
        pos_bboxes (list of dict): 每个正样本为一个字典，包含：
            - 'bbox': [x, y, w, h]
            - 'iou': float，匹配的最大IoU
    """
    purename = img_info['file_name'].split('.')[0]
    img_savedir = f'{crop_savedir}/{purename}'
    os.makedirs(img_savedir, exist_ok=True, mode=0o777)
    
    # 读取原图
    img_path = os.path.join(image_root, img_info['file_name'])
    image = Image.open(img_path)

    txt_lines,txt_bboxes = [],[]
    for label,bboxlist in enumerate([neg_bboxes, pos_bboxes]):
        for pidx, item in enumerate(bboxlist):
            x, y, w, h = item['bbox']
            prefix = 'neg' if label==0 else 'pos'
            crop_filename = f'{prefix}_{pidx}.png'
            txt_lines.append(f'{purename}/{crop_filename} {label}\n')
            # 1659bj010_0007/neg_81.png
            if purename != '1659bj010_0007' or pidx != 81:
                continue
            cropimg = image.crop([x, y, x+w, y+h])
            cropimg.save(f'{img_savedir}/{prefix}_{pidx}.png')
            item['crop_filename'] = crop_filename
            txt_bboxes.append(item)
    return txt_lines,txt_bboxes
        

def main():
    maxdet = 300
    assigner = MaxIoUAssigner(0.3, 0.3, 0.3)
    # for mode in ['fold1_train', 'fold1_val']:
    for mode in ['fold1_val']:
        jsonfile = f'data_resource/HMCHH/annofiles_roi/{mode}.json'
        coco_gt = COCO(jsonfile)

        with open(f'data_resource/HMCHH/proposals_file/{mode}_byarea.json', 'r', encoding='utf-8') as f:
            pred_info = json.load(f)
        
        imgid_to_preds = defaultdict(list)
        for pred in pred_info:
            imgid_to_preds[pred['image_id']].append(pred['bbox'])

        visual_cnt = 0
        mode_txtlines = []
        mode_bboxes = {}
        for img_id, predbboxes in tqdm(imgid_to_preds.items(), ncols=80):
        # for img_id, predbboxes in imgid_to_preds.items():
            # if visual_cnt > 50:
            #     break
            img_info = coco_gt.loadImgs([img_id])[0]
            ann_ids = coco_gt.getAnnIds(imgIds=[img_id])
            anns = coco_gt.loadAnns(ann_ids)

            pred_bboxes = torch.tensor(predbboxes)
            areas = pred_bboxes[:, 2] * pred_bboxes[:, 3]
            sorted_indices = torch.argsort(areas, descending=True)
            proposal_bboxes = pred_bboxes[sorted_indices][:maxdet]
            pred_instances = InstanceData()
            pred_instances.priors = torch.Tensor([[x,y,x+w,y+h] for x,y,w,h in proposal_bboxes])
            
            gt_instances = InstanceData()
            gt_bboxes, gt_labels = [],[]
            for ann in anns:
                x,y,w,h = ann['bbox']
                gt_bboxes.append([x,y,x+w,y+h])
                gt_labels.append(ann['category_id'])
            gt_instances.bboxes = torch.Tensor(gt_bboxes)
            gt_instances.labels = torch.Tensor(gt_labels).to(torch.long)

            assign_result = assigner.assign(pred_instances, gt_instances)

            # 提取对应预测框和 IoU
            pos_bboxes,neg_bboxes = [],[]
            for idx,gtidx in enumerate(assign_result.gt_inds):
                bbox = pred_instances.priors[idx].numpy().tolist()
                x1,y1,x2,y2 = bbox
                iou = assign_result.max_overlaps[idx].item()
                bboxitem = {
                    "bbox": [x1,y1,x2-x1,y2-y1],
                    "iou": iou
                }
                if gtidx > 0:
                    pos_bboxes.append(bboxitem)
                else:
                    neg_bboxes.append(bboxitem)
            
            # visualize_positive_matches(img_info, anns, pos_bboxes)
            # visual_cnt += 1
            
            txtlines,txt_bboxes = crop_save_matches(img_info, pos_bboxes, neg_bboxes)
            mode_txtlines.extend(txtlines)
            mode_bboxes[img_id] = txt_bboxes
        
        # with open(f'{clsfiles_savedir}/{mode}_bboxes.json', 'w', encoding='utf-8') as f:
        #     json.dump(mode_bboxes, f, ensure_ascii=False)
        # with open(f'{clsfiles_savedir}/{mode}.txt', 'w') as f:
        #     f.writelines(mode_txtlines)

def train_dataset_balance():
    with open(f'{clsfiles_savedir}/fold1_train.txt', 'r') as f:
        total_lines = f.readlines()
    neg_maxcnt = 10
    new_lines = []
    imgid_to_samples = defaultdict(list)
    for line in tqdm(total_lines, ncols=80):
        # 1655bj006_0008/neg_0.png 0
        path, label = line.strip().split(' ')
        label = int(label)
        if label == 1:
            new_lines.append(line)
        else:
            imgid = path.split('/')[0]  # 提取 imgid
            imgid_to_samples[imgid].append(line)
    pos_cnt = len(new_lines)
    # 每个 imgid 下最多随机保留 neg_maxcnt 个阴性样本
    for imgid, samples in imgid_to_samples.items():
        if len(samples) > neg_maxcnt:
            samples = random.sample(samples, neg_maxcnt)
        new_lines.extend(samples)
    neg_cnt = len(new_lines) - pos_cnt
    # 保存结果
    save_path = os.path.join(clsfiles_savedir, f'fold1_train_s{neg_maxcnt}.txt')
    with open(save_path, 'w') as f:
        random.shuffle(new_lines)
        f.writelines(new_lines)
    
    print(f'Origin: neg({len(total_lines)-pos_cnt}) pos({pos_cnt}) total({len(total_lines)})')
    print(f'Sampled: neg({neg_cnt}) pos({pos_cnt}) total({len(new_lines)})')

if __name__ == "__main__":
    clsfiles_savedir = 'data_resource/HMCHH/proposals_file'
    image_root = 'data_resource/HMCHH/JPEGImages'
    visual_savedir = 'statistic_results/HMCHH/matched_result_byarea'
    os.makedirs(visual_savedir, exist_ok=True, mode=0o777)
    crop_savedir = f'data_resource/HMCHH/proposal_crops'
    os.makedirs(crop_savedir, exist_ok=True, mode=0o777)
    # main()

    train_dataset_balance()

'''
neg_maxcnt = 20
Origin: neg(1398981) pos(13906) total(1412887)
Sampled: neg(111540) pos(13906) total(125446)

neg_maxcnt = 10
Origin: neg(1398981) pos(13906) total(1412887)
Sampled: neg(55770) pos(13906) total(69676)
'''