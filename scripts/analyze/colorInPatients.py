
import os
import random
import pandas as pd
from tqdm import tqdm
from cerwsi.utils import KFBSlide
from PIL import Image
import glob
import cv2
from sklearn.cluster import KMeans
import numpy as np
import shutil


def main():
    df_puretrain = pd.read_csv('data_resource/0511/4_pure_train.csv')
    df_jfswtrain = pd.read_csv('data_resource/0511/5_jfsw_train.csv')
    df_train_pos = pd.concat([
        df_puretrain[df_puretrain['kfb_clsid']==1],
        df_jfswtrain[df_jfswtrain['kfb_clsid']==1]
    ])
    df_val = pd.read_csv('data_resource/0511/6_val.csv')
    df_val_pos = df_val[df_val['kfb_clsid']==1]
    
    for mode,df_data in zip(['train','val'], [df_train_pos, df_val_pos]):
        savedir = f'data_resource/0511/slide_thumbnail/{mode}'
        os.makedirs(savedir, exist_ok=True, mode=0o777)
        
        for row in tqdm(df_data.itertuples(index=True), total=len(df_data), ncols=80):
            if row.Index not in [16,17,18]:
                continue
            slide = KFBSlide(row.kfb_path)
            LEVEL = 2
            width, height = slide.level_dimensions[LEVEL]
            cx,cy = width/2, height/2
            clip_w,clip_h = 2000,2000
            location = cx-(clip_w/2), cy-(clip_h/2)
            size = (clip_w,clip_h)
            read_result = Image.fromarray(slide.read_region(location, LEVEL, size))
            read_result.save(f'{savedir}/{row.patientId}.png')

def extract_color_hist(image_path, bins=(10, 10, 10)):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def extract_dominant_color(image_path, k=3):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(image)
    counts = np.bincount(kmeans.labels_)
    dominant = kmeans.cluster_centers_[np.argmax(counts)]
    return dominant  # shape: (3,)

def load_thumbnail(image_path, size=(128, 128)):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, size)
    return image

def cluster_and_save_grids(mode='hist'):
    assert mode in ['hist', 'dominant'], "mode must be 'hist' or 'dominant'"
    print(f"Clustering using mode: {mode}")

    n_clusters=10
    train_paths = glob.glob('data_resource/0511/slide_thumbnail/train/*.png')

    # 提取特征
    features = []
    for p in tqdm(train_paths, desc='Extracting features'):
        if mode == 'hist':
            feat = extract_color_hist(p)
        else:  # mode == 'dominant'
            feat = extract_dominant_color(p)
        features.append(feat)

    # 聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(features)

    # 分组图像路径
    clustered_paths = {i: [] for i in range(n_clusters)}
    for path, label in zip(train_paths, labels):
        clustered_paths[label].append(path)

    # 保存每个聚类的拼图
    outdir = f"data_resource/0511/slide_thumbnail/cluster_center/{mode}"
    os.makedirs(outdir, exist_ok=True)

    for cluster_id in range(n_clusters):
        paths = clustered_paths[cluster_id][:64]
        thumbs = [load_thumbnail(p) for p in paths]
        if not thumbs:
            continue

        rows, cols = 8, 8
        h, w, _ = thumbs[0].shape
        canvas = np.ones((rows * h, cols * w, 3), dtype=np.uint8) * 255

        for i, img in enumerate(thumbs):
            r, c = divmod(i, cols)
            canvas[r*h:(r+1)*h, c*w:(c+1)*w] = img

        save_path = os.path.join(outdir, f'cluster_{cluster_id}.png')
        cv2.imwrite(save_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

def find_most_representative_images():
    train_paths = glob.glob('data_resource/0511/slide_thumbnail/train/*.png')
    features = []
    n_clusters = 10
    for p in tqdm(train_paths, desc='Extracting features'):
        feat = extract_color_hist(p)
        features.append(feat)

    # 聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(features)

    features = np.array(features)
    centers = kmeans.cluster_centers_

    target_savedir = 'data_resource/0511/slide_thumbnail/target'
    os.makedirs(target_savedir, exist_ok=True, mode=0o777)

    for i in range(kmeans.n_clusters):
        idxs = np.where(labels == i)[0]
        cluster_feats = features[idxs]
        center = centers[i].reshape(1, -1)

        # 找最近距离的图像索引
        closest_idx = np.argmin(np.linalg.norm(cluster_feats - center, axis=1))
        representative_path = train_paths[idxs[closest_idx]]
        filename = os.path.basename(representative_path)
        image = cv2.imread(representative_path)
        h, w, _ = image.shape
        top = random.randint(0, h - 1000)
        left = random.randint(0, w - 1000)
        cropped = image[top:top + 1000, left:left + 1000]
        cv2.imwrite(f'{target_savedir}/{filename}', cropped)
        # shutil.copy(representative_path, f'{target_savedir}/{filename}')
        # print(f"Cluster {i}: {representative_path}")

if __name__ == "__main__":
    # main()

    # cluster_and_save_grids(mode='hist')      # 用颜色直方图
    # cluster_and_save_grids(mode='dominant')  # 用主色调

    find_most_representative_images()
