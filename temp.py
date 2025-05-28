import glob
import shutil
import os

from tqdm import tqdm

for imgpath in tqdm(glob.glob('data_resource/0511/WINDOW_SIZE_750/images/neg_slide_jfsw/*.png'), ncols=80):
    filename = os.path.basename(imgpath)
    descpath = f'data_resource/0511/WINDOW_SIZE_750/images/neg/{filename}'
    shutil.copy(imgpath, descpath)
