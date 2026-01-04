import os
import cv2
import matplotlib.pyplot as plt
from skimage.color import rgb2hed

def visual_contrast(root_dir, demo_list):
    save_dir = 'statistic_results/color_space/contrast'
    os.makedirs(save_dir, exist_ok=True, mode=0o777)

    for filename in demo_list:
        img_path = os.path.join(root_dir, filename)
        img_bgr = cv2.imread(img_path)

        # 颜色空间转换
        # BGR -> RGB (用于正常显示)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # BGR -> Lab (亮度L + 颜色分量a,b)
        img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
        # BGR -> HSV (色调H + 饱和度S + 亮度V)
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # RGB 子图
        axes[0].imshow(img_rgb)
        axes[0].set_title("RGB Space")
        axes[0].axis('off') # 关闭坐标轴显示

        # Lab 子图 (直接imshow显示的是Lab数值映射后的伪彩色)
        axes[1].imshow(img_lab)
        axes[1].set_title("Lab Space")
        axes[1].axis('off')

        # HSV 子图 (直接imshow显示的是HSV数值映射后的伪彩色)
        axes[2].imshow(img_hsv)
        axes[2].set_title("HSV Space")
        axes[2].axis('off')

        save_path = os.path.join(save_dir, filename)
        plt.tight_layout(pad=2.0) # 调整布局防止重叠
        plt.savefig(save_path, dpi=150) # 保存
        plt.close() # 关闭画布释放内存
        
        print(f"已保存: {save_path}")

    print("所有图片处理完成。")

def visual_HSV(root_dir, demo_list):
    save_dir = 'statistic_results/color_space/HSV'
    os.makedirs(save_dir, exist_ok=True, mode=0o777)

    for filename in demo_list:
        img_path = os.path.join(root_dir, filename)
        img_bgr = cv2.imread(img_path)

        # 颜色空间转换
        # BGR -> RGB (用于正常显示)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # BGR -> HSV (色调H + 饱和度S + 亮度V)
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(img_hsv) # 拆分通道

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        # 原图 RGB
        axes[0].imshow(img_rgb)
        axes[0].set_title("Original RGB")
        axes[0].axis('off')
        # Hue (色调) - 你的背景噪点主要会出现在这里
        axes[1].imshow(h, cmap='gray') 
        axes[1].set_title("H Channel (Hue)")
        axes[1].axis('off')
        # Saturation (饱和度) - 细胞核分割通常重点看这里
        axes[2].imshow(s, cmap='gray')
        axes[2].set_title("S Channel (Saturation)")
        axes[2].axis('off')
        # Value (亮度) - 类似黑白照片
        axes[3].imshow(v, cmap='gray')
        axes[3].set_title("V Channel (Value)")
        axes[3].axis('off')
        # 3. 保存
        save_path = os.path.join(save_dir, f"HSV_{filename}")
        plt.tight_layout(pad=2.0)
        plt.savefig(save_path, dpi=100)
        plt.close()
        print(f"已保存 HSV 分析: {filename}")

def visual_Lab(root_dir, demo_list):
    save_dir = 'statistic_results/color_space/Lab'
    os.makedirs(save_dir, exist_ok=True, mode=0o777)

    for filename in demo_list:
        img_path = os.path.join(root_dir, filename)
        img_bgr = cv2.imread(img_path)

        # 颜色空间转换
        # BGR -> RGB (用于正常显示)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
        l, a, b = cv2.split(img_lab) # 拆分通道

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        # 原图 RGB
        axes[0].imshow(img_rgb)
        axes[0].set_title("Original RGB")
        axes[0].axis('off')
        # L (Lightness) - 亮度，包含主要纹理细节
        axes[1].imshow(l, cmap='gray')
        axes[1].set_title("L Channel (Lightness)")
        axes[1].axis('off')
        # a (Green-Red) - 也就是红绿分量，你的细胞质/核对比可能在这里较强
        axes[2].imshow(a, cmap='gray')
        axes[2].set_title("a Channel (Green-Red)")
        axes[2].axis('off')
        # b (Blue-Yellow) - 蓝黄分量
        axes[3].imshow(b, cmap='gray')
        axes[3].set_title("b Channel (Blue-Yellow)")
        axes[3].axis('off')

        save_path = os.path.join(save_dir, f"Lab_{filename}")
        plt.tight_layout(pad=2.0)
        plt.savefig(save_path, dpi=100)
        plt.close()
        print(f"已保存 Lab 分析: {filename}")

def visual_HED(root_dir, demo_list):
    save_dir = 'statistic_results/color_space/HED'
    os.makedirs(save_dir, exist_ok=True, mode=0o777)

    for filename in demo_list:
        img_path = os.path.join(root_dir, filename)
        # 1. 读取 (skimage 需要 RGB)
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
        # 2. 核心：颜色反卷积 -> HED 空间
        # 通道 0: Hematoxylin (苏木精 - 染细胞核 - 深紫)
        # 通道 1: Eosin (伊红 - 染细胞质/基质 - 粉红)
        # 通道 2: DAB (通常用于免疫组化，在普通巴氏染色中通常是杂质或留空)
        img_hed = rgb2hed(img_rgb)
        # 3. 可视化
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(img_rgb)
        axes[0].set_title("Original RGB")
        # H 通道：专门看细胞核！
        axes[1].imshow(img_hed[:, :, 0], cmap='gray')
        axes[1].set_title("Hematoxylin (Nucleus)")
        # E 通道：专门看细胞质
        axes[2].imshow(img_hed[:, :, 1], cmap='gray')
        axes[2].set_title("Eosin (Cytoplasm)")
        # D 通道
        axes[3].imshow(img_hed[:, :, 2], cmap='gray')
        axes[3].set_title("DAB (Residual)")
        save_path = os.path.join(save_dir, f"Lab_{filename}")
        plt.tight_layout(pad=2.0)
        plt.savefig(save_path, dpi=100)
        plt.close()
        print(f"已保存 HED 分析: {filename}")



def main():
    root_dir = '/c23030/zly/datasets/CervicalDatasets/WINDOW_SIZE_1200/images/total_pos'
    demo_list = [
        'ZY_ONLINE_1_1475_2729992602050_251.png',
        'ZY_ONLINE_1_1467_3109169951868_48.png',
        'ZY_ONLINE_1_1463_1364285537399_45.png',
        'ZY_ONLINE_1_1452_1474860931029_0.png',
        'ZY_ONLINE_1_196_6918732866314_44.png',
        'ZY_ONLINE_1_148_2157410120226_11.png',
        'WXL_3_622_1494012159100_11.png',
        'WXL_1_221_2232300446470_56.png',
        'JFSW_2_1630_1523437965366_4.png',
        'JFSW_2_1572_2498492387100_0.png'
    ]

    # visual_contrast(root_dir, demo_list)
    # visual_HSV(root_dir, demo_list)
    # visual_Lab(root_dir, demo_list)
    visual_HED(root_dir, demo_list)

    

if __name__ == "__main__":
    main()