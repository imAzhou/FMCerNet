# import os
# from cerwsi.utils import KFBSlide,kfbslide_get_associated_image_names,kfbslide_read_associated_image
# import openslide

# svs_path = '/nfs-medical3/data/浙一胃HE与Fish数据/胃癌FISH片/2025126768-FISH-HE-KL.svs'
# kfb_path = '/medical-data/data/cervix/JFSW_1109/HSIL/C202028855.kfb'
# kfbf_path = '/nfs-medical3/data/浙一胃HE与Fish数据/胃癌FISH片/2025126768-FISH-KL.kfbf'

# # source_path = kfbf_path
# # slide = KFBSlide(source_path)
# # swidth, sheight = slide.level_dimensions[0]
# # associated_images = kfbslide_get_associated_image_names(slide._osr)
# # if 'label' not in associated_images:
# #     print(f'{source_path} haven\'t label!')
# # else:
# #     filename = os.path.splitext(os.path.basename(source_path))[0]
# #     image = kfbslide_read_associated_image(slide._osr, 'label')
# #     output_path = f"{filename}.png"
# #     image.save(output_path, "PNG")

# slide = openslide.OpenSlide(svs_path)
# print(slide.associated_images.keys())


import torch

data = torch.load('data_resource/0630/WINDOW_SIZE_1200/slide_feat_detector/JFSW_1_2.pt')
print()
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pycocotools import mask as mask_utils

def visualize_segmentation_comparison(imgRGB, our_object_list, cellpose_object_list, save_path):
    """
    可视化原图与两组分割结果的比较。
    
    Args:
        imgpath (str): 原图路径。
        our_object_list (list[dict]): 第一组检测结果，每个元素包含 'segmentation'。
        cellpose_object_list (list[dict]): 第二组检测结果。
        save_path (str): 输出图片保存路径。
    """
    # 创建两张副本用于绘制边缘
    img_our = imgRGB.copy()
    img_cellpose = imgRGB.copy()

    def draw_segmentation_edges(target_img, object_list, color):
        """在 target_img 上绘制 segmentation 的边缘线"""
        for obj in object_list:
            rle = obj['segmentation']
            mask = mask_utils.decode(rle)
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(target_img, contours, -1, color, 2)

    # 绘制两组结果
    draw_segmentation_edges(img_our, our_object_list, color=(255, 0, 0))       # 红色线条
    draw_segmentation_edges(img_cellpose, cellpose_object_list, color=(0, 255, 0))  # 绿色线条

    # 绘制三行图像
    fig, axes = plt.subplots(3, 1, figsize=(8, 12))
    axes[0].imshow(imgRGB)
    axes[0].set_title("Original Image")
    axes[1].imshow(img_our)
    axes[1].set_title("Our Model Segmentation")
    axes[2].imshow(img_cellpose)
    axes[2].set_title("Cellpose Segmentation")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"可视化结果已保存到：{save_path}")
    
