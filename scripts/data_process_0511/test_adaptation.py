from skimage.color import rgb2lab
# 多 target 的 Macenko normalizer
from tiatoolbox.tools.stainnorm import MacenkoNormalizer
import glob
import cv2
import numpy as np
import torch
import os
from mmengine.config import Config
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from cerwsi.datasets import load_data
from cerwsi.nets import PatchClsNet
from cerwsi.utils import set_seed

TARGET_PATHS = glob.glob('data_resource/0511/slide_thumbnail/target/*.png')
TARGET_NORMALIZERS = []
TARGET_LAB_MEAN    = []          # (K,3) Lab 均值，用于快速匹配

for p in TARGET_PATHS:
    img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
    norm = MacenkoNormalizer(); norm.fit(img)
    TARGET_NORMALIZERS.append(norm)
    TARGET_LAB_MEAN.append(rgb2lab(img.reshape(-1,3)/255.).mean(0))

TARGET_LAB_MEAN = np.vstack(TARGET_LAB_MEAN).astype(np.float32)  # shape (K,3)


def pick_normalizer(patch_rgb: np.ndarray) -> MacenkoNormalizer:
    """对单个 patch 选 Lab 均值最邻近的 target normalizer"""
    lab = rgb2lab(patch_rgb.reshape(-1,3)/255.).mean(0)
    idx = np.linalg.norm(TARGET_LAB_MEAN - lab, axis=1).argmin()
    target_imgpath = TARGET_PATHS[int(idx)]
    return TARGET_NORMALIZERS[int(idx)],target_imgpath

def visualize_patch(bgr_tensor, norm_bgr_tensor, target_imgpath, save_path):
    """
    将原图 BGR Tensor、标准化后的 BGR Tensor 与参考图拼图保存。
    
    参数：
        - bgr_tensor: (3, H, W)，原始 BGR 图像张量
        - norm_bgr_tensor: (3, H, W)，标准化后 BGR 图像张量
        - target_imgpath: str，参考图路径
        - save_path: str，保存路径（包含文件名）
    """
    # 将 BGR Tensor 转为 RGB ndarray：(H, W, 3)
    def tensor_to_rgb(tensor):
        rgb = tensor[[2, 1, 0], :, :].permute(1, 2, 0).cpu().numpy()
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return rgb

    original_rgb = tensor_to_rgb(bgr_tensor)
    normalized_rgb = tensor_to_rgb(norm_bgr_tensor)
    target_rgb = np.array(Image.open(target_imgpath).convert("RGB"))

    # 可视化三列图
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].imshow(original_rgb)
    axs[0].set_title("Original")
    axs[0].axis("off")

    axs[1].imshow(target_rgb)
    axs[1].set_title("Target")
    axs[1].axis("off")

    axs[2].imshow(normalized_rgb)
    axs[2].set_title("Normalized")
    axs[2].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()

set_seed(1234)
device = torch.device(f'cuda:2')
cfg = Config.fromfile('log/WINDOW_SIZE_1000/smartccs_518_fusiontrain/config.py')
cfg.backbone_cfg['backbone_ckpt'] = None
cfg.instance_ckpt = None
model = PatchClsNet(cfg).to(device)
model.load_ckpt('log/WINDOW_SIZE_1000/smartccs_518_fusiontrain/checkpoints/best.pth')
valloader = load_data(cfg, ['val'])

for idx, data_batch in enumerate(tqdm(valloader, ncols=80)):
    inputs = []
    for bgrTensor,datasample in zip(data_batch['inputs'], data_batch['data_samples']):
        patch_rgb = bgrTensor[[2, 1, 0], :, :].permute(1,2,0).numpy()   # bgr2rgb -> ndarray: (H,W,3)
        # —— Macenko 动态标准化 ——
        mac,target_imgpath = pick_normalizer(patch_rgb)
        patch_rgb_norm = mac.transform(patch_rgb.copy())      # uint8 RGB, ndarray: (H,W,3)
        # RGB ndarray: (H,W,3) -> BGR Tensot: (3, H, W)
        patch_bgr_norm = np.transpose(patch_rgb_norm[:, :, ::-1], (2, 0, 1))
        norm_bgrTensor = torch.Tensor(patch_bgr_norm.copy())
        inputs.append(norm_bgrTensor)
        
        # filename = os.path.basename(datasample.img_path)
        # savepath = f'statistic_results/0511/stain_norm/{filename}'
        # visualize_patch(bgrTensor,norm_bgrTensor,target_imgpath, savepath)

    data_batch['inputs'] = torch.stack(inputs).to(device)

    with torch.no_grad():
        outputs = model(data_batch, 'val')
    model.classifier.evaluator.process(data_samples=[outputs], data_batch=None)
    
metrics = model.classifier.evaluator.evaluate(len(valloader.dataset))
print(metrics)

'''
positive_thr: 0.3
+--------+--------------+-----------------+-----------------+
|  AUC   | img_accuracy | img_sensitivity | img_specificity |
+--------+--------------+-----------------+-----------------+
| 0.6894 |    0.5048    |      0.8515     |      0.3518     |
+--------+--------------+-----------------+-----------------+

+--------------------------+
|     confusion matrix     |
+-----+------+------+------+
|     |  0   |  1   | sum  |
+-----+------+------+------+
|  0  | 2353 | 4335 | 6688 |
|  1  | 438  | 2512 | 2950 |
| sum | 2791 | 6847 | 9638 |
+-----+------+------+------+

positive_thr: 0.5
+--------+--------------+-----------------+-----------------+
|  AUC   | img_accuracy | img_sensitivity | img_specificity |
+--------+--------------+-----------------+-----------------+
| 0.6894 |    0.5733    |      0.7705     |      0.4862     |
+--------+--------------+-----------------+-----------------+

+--------------------------+
|     confusion matrix     |
+-----+------+------+------+
|     |  0   |  1   | sum  |
+-----+------+------+------+
|  0  | 3252 | 3436 | 6688 |
|  1  | 677  | 2273 | 2950 |
| sum | 3929 | 5709 | 9638 |
+-----+------+------+------+


(No Adaptation) positive_thr: 0.3
+--------+--------------+-----------------+-----------------+
|  AUC   | img_accuracy | img_sensitivity | img_specificity |
+--------+--------------+-----------------+-----------------+
| 0.7255 |    0.6995    |      0.5963     |      0.7451     |
+--------+--------------+-----------------+-----------------+

+--------------------------+
|     confusion matrix     |
+-----+------+------+------+
|     |  0   |  1   | sum  |
+-----+------+------+------+
|  0  | 4983 | 1705 | 6688 |
|  1  | 1191 | 1759 | 2950 |
| sum | 6174 | 3464 | 9638 |
+-----+------+------+------+
'''
