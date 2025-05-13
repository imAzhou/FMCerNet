import json
from tqdm import tqdm
import os
import glob
import shutil

for mode in ['val']:
    cnt = 0
    with open(f'data_resource/0410/annofiles/{mode}_v0410.json', 'r') as f:
        patch_list = json.load(f)
    for patchinfo in tqdm(patch_list, ncols=80):
        if '0403jfsw' in patchinfo['prefix'] and patchinfo['diagnose'] == 1:
            purename = patchinfo['filename'].split('.')[0]
            src_path = f'data_resource/0403/images/Pos/{purename}.png'
            target_path = f'data_resource/0410/0403jfsw/images/Pos/{purename}.png'
            try:
                shutil.copy(src_path, target_path)
                cnt += 1
            except:
                print(f'File not exist: {src_path}')
    print(f'mode: {mode}: {cnt}')    
