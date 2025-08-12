import json
import cv2
from tqdm import tqdm
import os

with open('data_resource/WINDOW_SIZE_850/hardsample_annofiles/multilable_hs_round1.json', 'r', encoding='utf-8') as f:
    json_data = json.load(f)

for pinfo in tqdm(json_data['data_list'][11570:], ncols=80):
    
    prefix = pinfo['img_path'].split('/')[0]
    if prefix != 'neg_slide_r1':
        continue

    imgpath = f'data_resource/WINDOW_SIZE_850/images/{pinfo["img_path"]}'
    result = cv2.imread(imgpath)
    if result is None:
        print(imgpath)