import glob
import json
import shutil
import os
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import openslide
from PIL import Image

wsi_path = glob.glob('/nfs-medical/vipa-medical/zheyi/zly/NILM中霉菌滴虫线索细胞/2025-06-03/**.svs')
demo_path = wsi_path[0]
# slide = openslide.OpenSlide(demo_path)
slide = openslide.open_slide(demo_path)
print("Level count:", slide.level_count)
print("Level dimensions:", slide.level_dimensions)
print("Level downsamples:", slide.level_downsamples)

