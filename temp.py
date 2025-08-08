from PIL import Image
import pandas as pd
from cerwsi.utils import KFBSlide


df_pure = pd.read_csv('data_resource/0630/4_pure_train.csv')
df_jfsw = pd.read_csv('data_resource/0630/5_jfsw_train.csv')
df_test = pd.read_csv('data_resource/0630/7_test.csv')
df = pd.concat([df_pure, df_jfsw, df_test])

for pid in ['ZY_ONLINE_1_1763','ZY_ONLINE_1_1764','ZY_ONLINE_1_1765','ZY_ONLINE_1_1766']:
    rowinfo = df[df['patientId'] == pid].iloc[0]
    slide = KFBSlide(rowinfo.kfb_path)
    LEVEL = len(slide.level_dimensions) - 1 
    max_x, max_y = slide.level_dimensions[LEVEL]
    read_result = Image.fromarray(slide.read_region((0,0), LEVEL, (max_x,max_y)))
    read_result.save(f'{pid}.png')

