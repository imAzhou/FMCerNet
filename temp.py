import glob
import shutil
import os

from tqdm import tqdm

for imgpath in tqdm(glob.glob('data_resource/0511/WINDOW_SIZE_750/patch_inst_mask_jfsw/*.npz'), ncols=80):
    filename = os.path.basename(imgpath)
    descpath = f'data_resource/0511/WINDOW_SIZE_750/patch_inst_mask/{filename}'
    shutil.copy(imgpath, descpath)
