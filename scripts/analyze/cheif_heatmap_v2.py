import torch
import os
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from mmengine.config import Config
from pycocotools.coco import COCO
from cerwsi.datasets import load_data
from cerwsi.nets import PatchClsNet
from cerwsi.utils import set_seed
import cv2
import torch.nn.functional as F
import math
import numpy as np

parser = argparse.ArgumentParser()
# base args
parser.add_argument('config_file', type=str)
parser.add_argument('ckpt', type=str)
parser.add_argument('save_dir', type=str)
parser.add_argument('--seed', type=int, default=1234, help='random seed')

args = parser.parse_args()

def draw_heatmap(coco, imgpath, anns, prob_2d, pos_prob, save_path):
    image = cv2.imread(imgpath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_h, img_w = image.shape[:2]

    # 插值到图像大小
    prob_resized = F.interpolate(prob_2d.unsqueeze(0).unsqueeze(0), size=(img_h, img_w), mode='bilinear', align_corners=False).squeeze().cpu().numpy()

    image_with_box = image.copy()
    cat_id_to_name = {cat['id']: cat['name'] for cat in coco.loadCats(coco.getCatIds())}

    # 绘制bbox和类别标签
    for ann in anns:
        x, y, w, h = map(int, ann['bbox'])
        category_id = ann['category_id']
        label = cat_id_to_name.get(category_id, "unknown")

        cv2.rectangle(image_with_box, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
        cv2.putText(image_with_box, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # 绘图
    fig, axs = plt.subplots(1, 2, figsize=(12, 7))

    axs[0].imshow(image_with_box)
    axs[0].set_title("Image with BBoxes")
    axs[0].axis('off')

    axs[1].imshow(image_with_box)
    im2 = axs[1].imshow(prob_resized, cmap='jet', alpha=0.6)
    axs[1].set_title(f"Prob heatmap, pos_prob: {pos_prob:.3}")
    axs[1].axis('off')
    # fig.colorbar(im2, ax=axs[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def test_net(cfg, model, device):
    valloader = load_data(cfg, ['val'])
    coco = COCO(cfg.val_annojson)
    model.eval()
    attnmap_save_dir = f'{args.save_dir}/CHIEF_Heatmap'
    os.makedirs(attnmap_save_dir, exist_ok=True, mode=0o777)
    draw_cnt = 0

    for idx, data_batch in enumerate(tqdm(valloader, ncols=80)):
        if draw_cnt > 30:
            break
        with torch.no_grad():
            data_batch['inputs'] = data_batch['inputs'].to(device)
            outputs = model(data_batch, 'val')

        for datasample, patchprob, pos_prob in zip(data_batch['data_samples'], outputs['patch_probs'], outputs['img_probs']):
            image_id = datasample.img_id
            ann_ids = coco.getAnnIds(imgIds=[image_id])
            anns = coco.loadAnns(ann_ids)
            imgpath = datasample.img_path
            prob_2d = patchprob.squeeze(0)   # Tensor: (h,w)
            filename = os.path.basename(imgpath)
            save_path = f'{attnmap_save_dir}/{filename}'
            pred_label = (pos_prob>0.3).int()
            # if pred_label == 0 and datasample.diagnose == 1:
            draw_heatmap(coco, imgpath, anns, prob_2d, pos_prob.item(), save_path)
            draw_cnt += 1


def main():
    set_seed(args.seed)
    device = torch.device(f'cuda:0')

    cfg = Config.fromfile(args.config_file)
    cfg.save_result_dir = args.save_dir
    cfg.backbone_cfg['backbone_ckpt'] = None
    cfg.instance_ckpt = None
    model = PatchClsNet(cfg).to(device)
    model.load_ckpt(args.ckpt)
    test_net(cfg, model, device)


if __name__ == '__main__':
    main()

'''
python scripts/analyze/cheif_heatmap.py \
    log/WINDOW_SIZE_512/instance_onlychief/config.py \
    log/WINDOW_SIZE_512/instance_onlychief/checkpoints/best.pth \
    log/WINDOW_SIZE_512/instance_onlychief
'''