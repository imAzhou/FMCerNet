import glob
import json
import shutil
import os
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

wsi_path = glob.glob('/nfs-medical/vipa-medical/zheyi/zly/NILM中霉菌滴虫线索细胞/2025-06-03/**.svs')
data_list = []
for path in wsi_path:
    data_list.append([])