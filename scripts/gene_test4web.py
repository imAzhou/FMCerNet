import pandas as pd
import os
from tqdm import tqdm
import re
from natsort import natsorted
from cerwsi.utils import KFBSlide,kfbslide_get_associated_image_names,kfbslide_read_associated_image

def gene_csv():
    sample_nums = {
        'NILM': 100,
        'ASC-US': 50,
        'LSIL': 50,
        'ASC-H': 20,
        'HSIL': 19,
        'SCC': 4,
        'AGC-N': 2,
        'AGC': 2,
        'AGC-NOS': 2,
    }

    df_test = pd.read_csv('data_resource/0630/WINDOW_SIZE_1600/annofiles/67_0907_val.csv')
    df_test = df_test[df_test['kfb_source'].isin(['ZY_ONLINE_1','WXL_1','WXL_2','WXL_3'])]
    cls_stats = df_test['kfb_clsname'].value_counts().reset_index()
    cls_stats.columns = ['kfb_clsname', 'count']
    print(cls_stats)

    df_samples = []
    for clsname, n in sample_nums.items():
        df_cls = df_test[df_test['kfb_clsname'] == clsname]
        # 如果数量足够就随机抽样，不足就全取
        if len(df_cls) >= n:
            df_sampled = df_cls.sample(n=n, random_state=42)
        else:
            df_sampled = df_cls  # 不够的就全取
        df_samples.append(df_sampled)

    df_new = pd.concat(df_samples, ignore_index=True)
    df_new["pathologyId"] = ''
    save_path = "data_resource/0630/test4web.csv"
    df_new = df_new.iloc[natsorted(df_new.index, key=lambda i: df_new.at[i, 'patientId'])].reset_index(drop=True)
    df_new.to_csv(save_path, index=False)
    print(f"Saved sampled dataframe: {len(df_new)} rows to {save_path}")

def save_patholoid_img():
    save_path = "data_resource/0630/test4web.csv"
    output_dir = "data_resource/0630/test4web_patholoids"
    os.makedirs(output_dir, exist_ok=True, mode=0o777)
    df_data = pd.read_csv(save_path)
    
    patterns = [
        (r"^C\d{9}$", lambda m: m.group(0)),         # C开头 + 9个数字
        (r"^(C\d{9})001$", lambda m: m.group(1)),    # 上面模式 + 001，取前9位
        (r"^(ZC\d+)-CT$", lambda m: m.group(1)),     # ZC 开头 + 数字 + -CT
        (r"^(YC\d+)-\d+$", lambda m: m.group(1)),    # YC 开头 + 数字 + -数字
    ]

    for idx, row in tqdm(df_data.iterrows(), total=len(df_data), ncols=80):
        purename = os.path.basename(row.kfb_path).split('.')[0]
        pathoId = purename
        for pattern, handler in patterns:
            if match := re.match(pattern, purename):
                pathoId = str(handler(match))
                break
        df_data.at[idx, 'pathologyId'] = pathoId
        
        # slide = KFBSlide(row.kfb_path)
        # # 获取所有关联图像名称
        # associated_images = kfbslide_get_associated_image_names(slide._osr)
        # if 'label' not in associated_images:
        #     print(f'{row.patientId} haven\'t label!')
        #     continue
        # image = kfbslide_read_associated_image(slide._osr, 'label')
        # output_path = f"{output_dir}/{row.patientId}.png"
        # image.save(output_path, "PNG")
    df_data.to_csv(save_path, index=False)


if __name__ == "__main__":
    # gene_csv()
    save_patholoid_img()
