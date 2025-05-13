from tqdm import tqdm
from cerwsi.utils import KFBSlide
import pandas as pd
from PIL import Image

df_data = pd.read_csv('data_resource/zheyi_annofiles/0409_slide_anno.csv')
for row in tqdm(df_data.itertuples(index=False), total=len(df_data), ncols=80):
    
    slide = KFBSlide(row.kfb_path)
    swidth, sheight = slide.level_dimensions[0]
    x1,x2 = (swidth // 2) - 500, (swidth // 2) + 500
    y1 = sheight - 1000
    y2 = sheight - 100
    w,h = x2-x1, y2-y1
    read_img = Image.fromarray(slide.read_region((x1,y1), 0, (w,h)))
    read_img.save(f'{row.patientId}.png')
