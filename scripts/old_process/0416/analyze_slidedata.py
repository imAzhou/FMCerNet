from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from natsort import natsorted
from PIL import Image
from cerwsi.utils import KFBSlide,kfbslide_get_associated_image_names,kfbslide_read_associated_image

def img2pid():
    partial_pos = os.listdir('data_resource/0416/images/partial_pos')

    pId_imgs = defaultdict(list)
    for filename in tqdm(partial_pos, ncols=80):
        patientId = '_'.join(filename.split('_')[:3])
        pId_imgs[patientId].append(filename)
    
    arr = np.array([len(i) for i in pId_imgs.values()])
    print("均值:", np.mean(arr))
    print("方差:", np.var(arr))
    print("标准差:", np.std(arr))
    print("最小值:", np.min(arr))
    print("最大值:", np.max(arr))
    print("中位数:", np.median(arr))
    print("四分位数(Q1, Q2, Q3):", np.percentile(arr, [25, 50, 75]))

    return pId_imgs

def main(pId_imgs):
    data_root_dir = 'data_resource/0416/annofiles'
    for mode in ['train','val']:
        csv_file = f'{data_root_dir}/{mode}.csv'
        df_data = pd.read_csv(csv_file)
        few_anno_list = []
        for row in tqdm(df_data.itertuples(index=False),total=len(df_data), ncols=80):
            if row.kfb_clsid == 0 or row.kfb_source == 'ZY_ONLINE_1':
                continue
            if 'AGC' in row.kfb_clsname:
                continue
            valued_nums = len(pId_imgs[row.patientId])
            if valued_nums < 10:
                few_anno_list.append([valued_nums, row.patientId, row.kfb_clsname, row.kfb_path, row.kfb_source])
        df_few_anno_list = pd.DataFrame(few_anno_list, columns=['valued_nums', 'patientId', 'kfb_clsname', 'kfb_path', 'kfb_source'])
        df_few_anno_list = df_few_anno_list.sort_values(by='valued_nums', ascending=True)
        df_few_anno_list.to_csv(f'{save_dir}/{mode}.csv', index=False)

def extract_pathology_info():
    for mode in ['train','val']:
        csv_file = f'{save_dir}/{mode}.csv'
        df_data = pd.read_csv(csv_file)
        low_valued_extractinfo = []
        for row in tqdm(df_data.itertuples(index=False),total=len(df_data), ncols=80):
            slide = KFBSlide(row.kfb_path)
            # 获取所有关联图像名称
            associated_images = kfbslide_get_associated_image_names(slide._osr)
            if 'label' not in associated_images:
                print(f'No label!')
                continue
            image = kfbslide_read_associated_image(slide._osr, 'label')
            output_path = f'{label_save_dir}/{row.patientId}.png'
            image.save(output_path)

            image = kfbslide_read_associated_image(slide._osr, 'thumbnail')
            output_path = f'{thumbnail_save_dir}/{row.patientId}.png'
            image.save(output_path)

            low_valued_extractinfo.append([row.patientId, row.kfb_clsname, '', 0])
        df_low_valued_extractinfo = pd.DataFrame(low_valued_extractinfo, columns=['patientId', 'kfb_clsname', 'pathology_number', 'year'])
        df_low_valued_extractinfo = df_low_valued_extractinfo.loc[
            natsorted(df_low_valued_extractinfo.index, key=lambda i: df_low_valued_extractinfo.loc[i, 'patientId'])
        ]
        df_low_valued_extractinfo.to_csv(f'{save_dir}/extractinfo_{mode}.csv', index=False)

def filter_csv():
    anno_slide = []
    for mode in ['train','val']:
        new_mode_row = []
        df_data = pd.read_csv(f'{save_dir}/extractinfo_{mode}.csv')
        remove_pids = df_data[df_data['year'] == 0]['patientId'].tolist()
        anno_pids = df_data[df_data['year'] > 0]['patientId'].tolist()

        df_mode = pd.read_csv(f'data_resource/0416/annofiles/{mode}.csv')
        for row in tqdm(df_mode.itertuples(index=False), total=len(df_mode), ncols=80):
            if row.patientId not in remove_pids:
                new_mode_row.append(row)
            if row.patientId in anno_pids:
                anno_slide.append(row)
        df_new_mode = pd.DataFrame(new_mode_row, columns=df_mode.columns)
        df_new_mode.to_csv(f'data_resource/0416/annofiles/{mode}_0422.csv', index=False)
    
    df_anno_slide = pd.DataFrame(anno_slide, columns=df_mode.columns)  
    df_anno_slide.to_csv('data_resource/0416/annofiles/0422_slide_anno.csv', index=False)
    

if __name__ == "__main__":
    save_dir = 'statistic_results/0416/low_valued_nums'
    # os.makedirs(save_dir, exist_ok=True)
    # pId_imgs = img2pid()
    # main(pId_imgs)

    # label_save_dir = 'statistic_results/0416/pathology_info'
    # thumbnail_save_dir = 'statistic_results/0416/pathology_thumbnail'
    # os.makedirs(label_save_dir, exist_ok=True)
    # os.makedirs(thumbnail_save_dir, exist_ok=True)
    # extract_pathology_info()

    # 生成要给医生标注的csv，从train和val中删除剩下的，标注数量少的 slide
    filter_csv()