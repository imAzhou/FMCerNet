import glob
import shutil
import os

from tqdm import tqdm

for imgpath in tqdm(glob.glob('/c22073/zly/datasets/CervicalDatasets/LCerScanv1_750/images/neg_slide/*.png'), ncols=80):
    filename = os.path.basename(imgpath)
    descpath = f'/c22073/zly/datasets/CervicalDatasets/LCerScanv1_750/images/neg/{filename}'
    shutil.move(imgpath, descpath)
