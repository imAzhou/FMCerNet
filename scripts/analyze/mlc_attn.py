import torch
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=r"^xFormers is available \(.*\)")
from tqdm import tqdm
import argparse
from mmengine.config import Config
import mmengine.dist as dist
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch.nn.functional as F
from PIL import Image
import numpy as np
from cerwsi.datasets import load_data
from cerwsi.nets import PatchNet
from cerwsi.utils import set_seed, init_distributed_mode, is_main_process


parser = argparse.ArgumentParser()
# base args
parser.add_argument('config_file', type=str)
parser.add_argument('ckpt', type=str)
parser.add_argument('save_dir', type=str)
parser.add_argument('cocojsonfile', type=str, help='use to visual gt bboxes')
parser.add_argument('--val_json', type=str, help='assign val jsondatas')
parser.add_argument('--visual_nums', type=int, default=-1, help='-1 means visual all FN imgs')
parser.add_argument('--seed', type=int, default=1234, help='random seed')

args = parser.parse_args()

def test_net(args, cfg, model):
    valloader = load_data(cfg, ['val'])
    model.eval()
    pbar = valloader
    if is_main_process():
        pbar = tqdm(valloader, ncols=80)
    
    batch_outputs = []
    for idx, data_batch in enumerate(pbar):
        if idx > 2:
            break
        with torch.no_grad():
            outputs = model(data_batch, 'val')
        batch_outputs.extend([item.cpu() for item in outputs])

    results = dist.collect_results(batch_outputs, len(valloader.dataset), device='cpu')
    if is_main_process():
        pbar.close()
        img_savedir = f'{args.save_dir}/visual_attn'
        os.makedirs(img_savedir, exist_ok=True, mode=0o777)
        visual_attn(results, args.cocojsonfile, args.visual_nums, img_savedir)

def heatmap_attn(img, item, annos, coco, pred_labels_str, save_path):
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    num_tokens = item.attn.shape[0]
    h = w = int(np.sqrt(num_tokens))
    attnscore = F.softmax(item.attn, dim=0)
    # attnscore = item.attn
    attn2d = attnscore.reshape(h, w)
    w_img, h_img  = img.size
    attn_resized = F.interpolate(
        attn2d[None, None, :, :].float(),
        size=(h_img, w_img),
        mode='bilinear',
        align_corners=False
    )[0, 0].numpy()
    plt.imshow(attn_resized, cmap="jet", alpha=0.6)
    ax = plt.gca()
    for ann in annos:
        bbox = ann['bbox']  # [x,y,w,h]
        cls_id = ann['category_id']
        cls_name = coco.cats[cls_id]['name']
        color = coco.cats[cls_id]['color'] if 'color' in coco.cats[cls_id] else [255, 0, 0]
        rect = patches.Rectangle(
            (bbox[0], bbox[1]), bbox[2], bbox[3],
            linewidth=2, edgecolor=[c/255 for c in color], facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(
            bbox[0], bbox[1] - 2, cls_name,
            fontsize=6, color=[c/255 for c in color]
        )

    plt.title(pred_labels_str)
    
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

def heatmap_ncls_attn(img, item, annos, coco, pred_labels_str, save_path, classes):
    # visual attn, item.attn shape is # (n_cls, num_tokens)
    vmin = item.attn.min()
    vmax = item.attn.max()
    # vmin = 0
    # vmax = 1

    # attn_sum = item.attn.sum(dim=0, keepdim=True)          # (1, num_tokens)
    # attn_maps = torch.cat([attn_sum, item.attn], dim=0)     # (n_cls+1, num_tokens)
    attn_maps = item.attn     # (n_cls, num_tokens)
    n_maps, num_tokens = attn_maps.shape
    h = w = int(np.sqrt(num_tokens))
    attn_maps = attn_maps.reshape(n_maps, h, w)
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    titles = [pred_labels_str] + [classes[i] for i in range(n_maps)]
    for i, ax in enumerate(axes.ravel()):
        if i == 0:  # 第一张图显示 原图 + gt_bbox
            for ann in annos:
                bbox = ann['bbox']  # [x,y,w,h]
                cls_id = ann['category_id']
                cls_name = coco.cats[cls_id]['name']
                color = coco.cats[cls_id]['color'] if 'color' in coco.cats[cls_id] else [255, 0, 0]
                rect = patches.Rectangle(
                    (bbox[0], bbox[1]), bbox[2], bbox[3],
                    linewidth=2, edgecolor=[c/255 for c in color], facecolor='none'
                )
                ax.add_patch(rect)
                ax.text(
                    bbox[0], bbox[1] - 2, cls_name,
                    fontsize=6, color=[c/255 for c in color]
                )
            ax.imshow(img)
            ax.set_title(titles[i])
        elif i < n_maps+1:
            # ax.imshow(attn_maps[i].numpy(), cmap='jet', alpha=0.6)
            ax.imshow(img)
            w_img, h_img  = img.size
            attn_resized = F.interpolate(
                attn_maps[i-1][None, None, :, :].float(),
                size=(h_img, w_img),
                mode='bilinear',
                align_corners=False
            )[0, 0].numpy()
            ax.imshow(attn_resized, cmap="jet", alpha=0.6, vmin=vmin, vmax=vmax)
            # ax.imshow(attn_resized, cmap="jet", alpha=0.6)
            ax.set_title(titles[i])
        
        ax.axis("off")
    
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

def main():
    init_distributed_mode(args)
    set_seed(args.seed)
    device = torch.device(f'cuda:{os.getenv("LOCAL_RANK")}')

    cfg = Config.fromfile(args.config_file)
    cfg.backbone_cfg['backbone_ckpt'] = None
    cfg.instance_ckpt = None
    if args.val_json:
        cfg.val_datasets['ann_file'] = args.val_json
    model = PatchNet(cfg).to(device)
    model.load_ckpt(args.ckpt)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    model = model.module
    os.makedirs(args.save_dir, exist_ok=True, mode=0o777)
    test_net(args, cfg, model)
    torch.distributed.destroy_process_group()

def visual_attn(pred_result, coco_jsonfile, visual_nums, img_savedir):
    thr = 0.5
    visual_cnt = 0
    coco = COCO(coco_jsonfile)
    classes = [i['name'] for i in coco.cats.values()]
    filename2imgid = {}
    for imgitem in tqdm(coco.imgs.values(), ncols=90, desc='Load filename2imgid'):
        filename = imgitem['file_name'].split('/')[-1]
        filename2imgid[filename] = imgitem['id']

    for item in tqdm(pred_result, ncols=90, desc='Visual attn images'):
        if visual_nums>0 and visual_cnt>=visual_nums:
            break
        gt_diagnose = int(len(item.gt_label)>0)
        pred_diagnose = int(item.img_prob > 0.5)
        pred_multi_label = [clsidx for clsidx,cls_score in enumerate(item.pos_prob) if cls_score > thr]
        TP_flag = gt_diagnose == 1 and pred_diagnose == 1 and len(pred_multi_label) > 0
        if not TP_flag:
            continue
        filename = os.path.basename(item.img_path)
        if filename not in filename2imgid:
            continue
        img = Image.open(item.img_path).convert("RGB")
        annids = coco.getAnnIds([filename2imgid[filename]])
        annos = coco.loadAnns(annids)
        save_path = os.path.join(img_savedir, filename)

        # pred_labels_str = f'prob: {item.img_prob:.2f}'
        pred_labels_str = f'Pos prob: {item.img_prob:.2f}, ' + (", ".join([classes[idx] for idx in pred_multi_label]) if len(pred_multi_label) > 0 else "None")
        heatmap_attn(img, item, annos, coco, pred_labels_str, save_path)

        # pred_multi_label = [clsidx for clsidx,cls_score in enumerate(item.pos_prob) if cls_score > thr]
        # pred_labels_str = ", ".join([classes[idx] for idx in pred_multi_label]) if len(pred_multi_label) > 0 else "None"
        # heatmap_ncls_attn(img, item, annos, coco, pred_labels_str, save_path, classes)

        visual_cnt += 1


if __name__ == '__main__':
    main()


'''
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=12341 scripts/analyze/mlc_attn.py \
    log/WS1600/wscernet/mlc_f1_27.35/config.py \
    log/WS1600/wscernet/mlc_f1_27.35/checkpoints/best.pth \
    log/WS1600/wscernet/mlc_f1_27.35 \
    data_resource/WINDOW_SIZE_1600/annofiles/val_noNeg_cocoformat.json \
    --visual_nums 50

data_resource/WINDOW_SIZE_1600/annofiles/puretrain_noNeg_cocoformat.json
data_resource/WINDOW_SIZE_1600/annofiles/val_noNeg_cocoformat.json
'''