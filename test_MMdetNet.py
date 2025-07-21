import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=r"^xFormers is available \(.*\)")
from tqdm import tqdm
import argparse
from mmengine.config import Config
from pycocotools.coco import COCO
import os
from mmdet.evaluation import CocoMetric
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
from mmengine.registry import init_default_scope
from cerwsi.utils import set_seed,is_bbox_inside
from torchvision import transforms
from mmdet.structures import DetDataSample
from mmdet.models.detectors import FasterRCNN,DINO

POSITIVE_CLASS = ['AGC', 'ASC-US','LSIL', 'ASC-H', 'HSIL', 'SCC']

parser = argparse.ArgumentParser()
# base args
parser.add_argument('config_file', type=str)
parser.add_argument('ckpt', type=str)
parser.add_argument('save_dir', type=str)
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--visual_nums', type=int, default=-1, help='visual sample nums')
parser.add_argument('--visual_save_dir', type=str)
parser.add_argument('--calc_metric', action='store_true')

args = parser.parse_args()

def visua_pred(img_path, gt_info, filtered_bboxes, filtered_scores, filtered_labels, savepath):
    """
    img_path: 原图路径
    gt_info: COCO 的 ann 格式列表，元素中含有 'bbox' 和 'category_id'
    filtered_bboxes: List of predicted bboxes (x1, y1, x2, y2)
    filtered_scores: List of prediction scores
    filtered_labels: List of prediction labels
    """
    img = Image.open(img_path).convert("RGB")

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(img)
    axes[0].set_title("Ground Truth")
    axes[1].imshow(img)
    axes[1].set_title("Predictions")

    # 可视化 Ground Truth
    for ann in gt_info:
        x, y, w, h = ann['bbox']
        cat_id = ann['category_id']
        label_str = POSITIVE_CLASS[cat_id-1]
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='green', facecolor='none')
        axes[0].add_patch(rect)
        axes[0].text(x, y - 5, label_str, color='green', fontsize=10, backgroundcolor='white')

    # 可视化预测框
    for (x1, y1, x2, y2), score, label in zip(filtered_bboxes, filtered_scores, filtered_labels):
        label_str = POSITIVE_CLASS[label]
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='red', facecolor='none')
        axes[1].add_patch(rect)
        axes[1].text(x1, y1 - 5, f'{label_str} {score:.2f}', color='red', fontsize=10, backgroundcolor='white')

    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()

def postprocess_pred(pred_bboxes,pred_scores,pred_labels):
    filtered_bboxes,filtered_scores,filtered_labels = [],[],[]
    for bbox,score,label in zip(pred_bboxes,pred_scores,pred_labels):
        if score > 0.2:
            filtered_bboxes.append(bbox.tolist()) # x1,y1,x2,y2
            filtered_scores.append(score.item())
            filtered_labels.append(label.item())
    if len(filtered_bboxes) == 0:
        return [],[],[]
    
    bboxes = np.array(filtered_bboxes)
    labels = np.array(filtered_labels)
    areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    sorted_indices = np.argsort(-areas)  # 从大到小排序

    n = len(bboxes)
    used = np.zeros(n, dtype=bool)
    final_bboxes,final_scores,final_labels = [],[],[]
    for i in sorted_indices:
        if used[i]:
            continue
        group_indices = [i]
        for j in sorted_indices:
            if i == j or used[j]:
                continue
            if is_bbox_inside(bboxes[j], bboxes[i], tolerance=5):
                group_indices.append(j)
                used[j] = True

        # 当前 group 最外层是 i，label 取 group 内最大值
        max_label = labels[group_indices].max()
        final_bboxes.append(bboxes[i].tolist())
        final_scores.append(filtered_scores[i])
        final_labels.append(int(max_label))
        used[i] = True

    return final_bboxes,final_scores,final_labels

def test_net(cfg, pn_model):
    inputsize = pn_model.img_size
    transform = transforms.Compose([
        transforms.Resize(inputsize),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    coco = COCO(cfg.val_annojson)
    if args.calc_metric:
        coco_metric = CocoMetric(
            ann_file=cfg.val_annojson,
            metric='proposal',
            classwise=False,
            iou_thrs=[0.3],
            proposal_nums=(100, 300, 1000)
        )
        coco_metric.dataset_meta = dict(classes=POSITIVE_CLASS)
    
    visual_cnt = 0
    for item in tqdm(coco.imgs.values(), ncols=80):
        purename = item["file_name"].split('/')[-1]
        img_path = f'{cfg.img_dir}/{item["file_name"]}'
        img_tensor = transform(Image.open(img_path))
        img_tensor = img_tensor.unsqueeze(0).to(pn_model.device)
        data_batch = dict(
            inputs=img_tensor,
            data_samples = [DetDataSample(
                metainfo={
                    'img_shape':(inputsize,inputsize),
                    'ori_shape':(item['width'],item['height']),
                    'scale_factor': (inputsize/item['width'], inputsize/item['height'])
                },
                batch_input_shape=(inputsize,inputsize),
            )]
        )
        with torch.no_grad():
            outputs = pn_model(data_batch['inputs'], data_batch['data_samples'], mode="predict")
        predresult = outputs[0].pred_instances
        pred_bboxes,pred_scores,pred_labels = predresult.bboxes.cpu(), predresult.scores.cpu(), predresult.labels.cpu()

        filtered_bboxes, filtered_scores, filtered_labels = postprocess_pred(pred_bboxes,pred_scores,pred_labels)
        if args.visual_nums > 0 and visual_cnt < args.visual_nums:
            gt_info = coco.loadAnns(coco.getAnnIds(imgIds=[item['id']]))
            savedir = args.visual_save_dir
            os.makedirs(savedir, exist_ok=True, mode=0o777)
            visua_pred(img_path, gt_info, filtered_bboxes, filtered_scores, filtered_labels, f'{savedir}/{purename}')
            visual_cnt += 1
        
        if args.calc_metric:
            pred_instances = dict(bboxes=pred_bboxes,scores=pred_scores,labels=pred_labels,)
            coco_metric.process(
                {},
                [dict(pred_instances=pred_instances, 
                    img_id=item['id'], ori_shape=(item['width'], item['height']))])
    
    if args.calc_metric:
        print(f'Eval Result:')
        eval_results = coco_metric.evaluate(size=len(coco.imgs.values()))
        print(eval_results)
    

def main():
    set_seed(args.seed)
    device = torch.device('cuda:0')
    cfg = Config.fromfile(args.config_file)
    del cfg.model.type

    cfg.val_annojson = cfg.val_dataset.data_root + cfg.val_dataset.ann_file
    cfg.img_dir = cfg.val_dataset.data_root + cfg.val_dataset.data_prefix['img'][:-1]
    model = DINO(**cfg.model).to(device)
    model.img_size = cfg.input_size
    model.device = device
    ckpt = torch.load(args.ckpt,weights_only=False,map_location=device)['state_dict']
    model.load_state_dict(ckpt)
    model.eval()
    test_net(cfg, model)


if __name__ == '__main__':
    init_default_scope('mmdet')
    main()

'''
python test_MMdetNet.py \
    log/WS1600/vis_data/config.py \
    log/WS1600/epoch_8.pth \
    log/WS1600 \
    --visual_nums 50 \
    --visual_save_dir log/WS1600/pred_result
'''