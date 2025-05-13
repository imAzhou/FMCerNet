import pandas as pd
from tqdm import tqdm
import os
import shutil
import json
from PIL import Image
from cerwsi.utils import draw_OD,KFBSlide
import matplotlib.pyplot as plt
import numpy as np

LEVEL = 1
PATCH_EDGE = 700
POSITIVE_CLASS = ['ASC-US', 'LSIL', 'ASC-H', 'HSIL', 'SCC', 'AGC-NOS', 'AGC', 'AGC-N', 'AGC-FN']
colors = plt.cm.tab10(np.linspace(0, 1, len(POSITIVE_CLASS)))[:, :3] * 255
category_colors = {cat: tuple(map(int, color)) for cat, color in zip(POSITIVE_CLASS, colors)}

def vis_sample(slide,patchinfo):
    sample_save_dir = 'statistic_results/0319/diff_pos_sample'
    os.makedirs(sample_save_dir, exist_ok=True)
    x1,y1 = patchinfo['square_x1y1']
    innerbbox,bbox_clsname = patchinfo['bboxes'],patchinfo['clsnames']

    inside_items = []
    for coords,clsname in zip(innerbbox,bbox_clsname):
        inside_items.append({'sub_class': clsname,'region': coords})
    
    location, level, size = (x1,y1), LEVEL, (PATCH_EDGE,PATCH_EDGE)
    read_result = Image.fromarray(slide.read_region(location, level, size))
    filename = patchinfo["filename"]
    square_coords = [0,0,PATCH_EDGE,PATCH_EDGE]
    draw_OD(read_result, f'{sample_save_dir}/{filename}', square_coords, inside_items,category_colors)

root_dir = 'data_resource/0319/annofiles'
for mode in ['train', 'val']:
    with open(f'{root_dir}/{mode}_posslide_patches.json', 'r') as f:
        v1_data = json.load(f)
    with open(f'{root_dir}/{mode}_posslide_patches_v2.json', 'r') as f:
        v2_data = json.load(f)
    
    v2_filenames = []
    for slideinfo in v2_data:
        for patchinfo in slideinfo['patch_list']:
            v2_filenames.append(patchinfo['filename'])
    
    diff_patchinfo = []
    for slideinfo in v1_data:
        for patchinfo in slideinfo['patch_list']:
            if patchinfo['filename'] not in v2_filenames and patchinfo['diagnose'] == 1:
                slide = KFBSlide('/medical-data/data/' + slideinfo['kfb_path'])
                vis_sample(slide,patchinfo)
                diff_patchinfo.append(patchinfo)
    print()