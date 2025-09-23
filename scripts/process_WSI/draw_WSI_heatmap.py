import torch
import os
import warnings
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
warnings.filterwarnings("ignore", category=UserWarning)
from mmengine.config import Config
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from tqdm import tqdm
import cv2
from mmengine.registry import init_default_scope
from cerwsi.nets import PatchNet,ValidClsNet
from cerwsi.utils import set_seed
from cerwsi.utils.wsi_handler import WSIHandler


LEVEL,PATCH_EDGE = 0,1600
CERTAIN_THR,POSITIVE_THR = 0.7,0.3
SEED,SAFE_MARGIN = 1234,100
test_bs = 128
valid_ckpt = 'checkpoints/valid_cls_best.pth'
pnmodel_rootdir = 'log/WS1600/2025_08_26_13_29_55'
mmcls_config_file = f'{pnmodel_rootdir}/config.py'
mmcls_ckpt = f'{pnmodel_rootdir}/checkpoints/best.pth'
savedir = f'{pnmodel_rootdir}/WSI_heatmap/ZY_ONLINE_1_1467'
os.makedirs(savedir, exist_ok=True, mode=0o777)
kfb_path = '/nfs-medical/vipa-medical/zheyi/zly/KFBs/till_0318/HSIL/ZC202309458-CT.kfb'


def get_models(device):
    init_default_scope('mmpretrain')
    valid_model = ValidClsNet()
    valid_model.to(device)
    valid_model.device = device
    valid_model.eval()
    valid_model.load_state_dict(torch.load(valid_ckpt))

    cfg = Config.fromfile(mmcls_config_file)
    cfg.backbone_cfg['backbone_ckpt'] = None
    mlcls_model = PatchNet(cfg).to(device)
    mlcls_model.img_size = cfg.input_size
    mlcls_model.load_ckpt(mmcls_ckpt)
    mlcls_model.eval()

    return valid_model,mlcls_model

def set_heatmap(wsi_heatmap, downsample_ratio,patchinfo):
    # num_tokens = patchinfo['img_attnmap'].shape[0]
    # h = w = int(np.sqrt(num_tokens))
    # attn2d = patchinfo['img_attnmap'].reshape(h, w)
    # w_img, h_img  = PATCH_EDGE,PATCH_EDGE
    # attn_resized = F.interpolate(
    #     attn2d[None, None, :, :].float(),
    #     size=(h_img, w_img),
    #     mode='bilinear',
    #     align_corners=False
    # )[0, 0]

    x1,y1 = patchinfo['xy']
    x2,y2 = x1+PATCH_EDGE, y1+PATCH_EDGE
    x1, y1, x2, y2 = (
        np.array([x1, y1, x2, y2]) / downsample_ratio
    ).round().astype(int)
    wsi_heatmap[y1:y2,x1:x2] = patchinfo['img_prob']
    return wsi_heatmap

def collect_unique_xy(slide_patchlist, downsample_ratio):
    # 1. 收集缩放后的顶点坐标
    coords = np.array([item['xy'] for item in slide_patchlist], dtype=np.float32)
    coords = np.rint(coords / downsample_ratio).astype(int)  # 四舍五入转 int
    xs, ys = coords[:, 0], coords[:, 1]

    # 2. 找交集：哪些 x 值和 y 值出现 >=2 次
    unique_x = [x for x in np.unique(xs) if np.sum(xs == x) > 1]
    unique_y = [y for y in np.unique(ys) if np.sum(ys == y) > 1]

    return unique_x,unique_y

def draw_patch_lines(read_result, unique_x,unique_y, fill="black"):
    # 在 read_result 上绘制直线
    draw = ImageDraw.Draw(read_result)
    W, H = read_result.size
    for x in unique_x:
        draw.line([(x, 0), (x, H)], fill=fill, width=2)
    for y in unique_y:
        draw.line([(0, y), (W, y)], fill=fill, width=2)
    return read_result

def main():
    set_seed(SEED)
    device = torch.device('cuda:1')
    valid_model,mlcls_model = get_models(device)
    wsi_handler = WSIHandler(kfb_path, PATCH_EDGE, LEVEL, 
                                 certain_thr=CERTAIN_THR, positive_thr=POSITIVE_THR)
    slide_patchlist = wsi_handler.init_patchlist({
        'image': None,
        'valid_prob': 0, 
        'valid_flag': -1,
        'img_prob': 0, 
        'pred_label': -1,
        'img_attnmap': None     #   (num_tokens,)
    })
    smallest_level = len(wsi_handler.slide.level_downsamples)-1
    downsample_ratio = wsi_handler.slide.level_downsamples[smallest_level]
    smallest_width, smallest_height = wsi_handler.slide.level_dimensions[smallest_level]
    wsi_heatmap = torch.zeros((smallest_height,smallest_width))

    valid_datapool, mlcls_datapool = [],[]
    for p_idx,patchinfo in enumerate(tqdm(slide_patchlist, ncols=80)):
        img_input,_ = wsi_handler.read_cv2img(patchinfo['xy'])
        patchinfo['image'] = img_input
        valid_datapool.append(patchinfo)
        if len(valid_datapool) % test_bs == 0 or p_idx == len(slide_patchlist)-1:
            wsi_handler.infer_valid_fn(valid_model, valid_datapool)
            mlcls_datapool = [item for item in valid_datapool if item['valid_flag']==2]
            if len(mlcls_datapool) > 0:
                wsi_handler.infer_pn_fn(mlcls_model, mlcls_datapool)
            for item in valid_datapool:
                if item['pred_label'] == 1:
                    wsi_heatmap = set_heatmap(wsi_heatmap, downsample_ratio, item)
                del item['image']
            torch.cuda.empty_cache()
            valid_datapool, mlcls_datapool = [],[]
    
    read_result = wsi_handler.save_thumbnail(f'{savedir}/thumbnail.png')
    unique_x,unique_y = collect_unique_xy(slide_patchlist, downsample_ratio)
    read_result_withlines = draw_patch_lines(read_result, unique_x,unique_y)
    read_result_withlines.save(f'{savedir}/thumbnail_withlines.png')
    plt.figure(figsize=(6, 6))
    plt.imshow(read_result)

    # token_h = round(smallest_height / (PATCH_EDGE / downsample_ratio))
    # token_w = round(smallest_width / (PATCH_EDGE / downsample_ratio))
    # wsi_heatmap_resized = cv2.resize(wsi_heatmap.cpu().numpy(), (token_w, token_h), interpolation=cv2.INTER_NEAREST)
    # img_heatmap_smooth = cv2.resize(wsi_heatmap_resized, (smallest_width, smallest_height), interpolation=cv2.INTER_LINEAR)
    # plt.imshow(img_heatmap_smooth, cmap="jet", alpha=0.6)

    plt.imshow(wsi_heatmap, cmap="jet", alpha=0.4)
    # 在 plt 上绘制直线
    for x in unique_x:
        plt.axvline(x=x, color="black", linewidth=0.5)  # 垂直线
    for y in unique_y:
        plt.axhline(y=y, color="black", linewidth=0.5)  # 水平线

    plt.axis("off")
    plt.savefig(f'{savedir}/heatmap.png', bbox_inches='tight', dpi=150)
    plt.close()
    


if __name__ == "__main__":
    main()
