import glob
import json
import shutil
import os
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

save_dir = 'data_resource/0511'
df_abandon = pd.read_csv('data_resource/group_csv/abandon.csv')
df_partial_pos = pd.read_csv(f'{save_dir}/1_jfsw_pos.csv')
df_total_pos = pd.read_csv(f'{save_dir}/0_zheyi_pos.csv')
df_noann_pos = pd.read_csv(f'{save_dir}/2_noann_pos.csv')

with open('data_resource/zheyi_annofiles/宫颈液基细胞—RoI.json', 'r', encoding='utf-8') as f:
    group_json_data = json.load(f)

cnt = defaultdict(int)
empty_roi = 0
for item in group_json_data['Group2']:
    patientId = '_'.join(item['imageName'].split('_')[:3])
    annitems = item['annotations'][0]['annotationResult']
    if len(annitems) == 0:
        empty_roi += 1
print(empty_roi)


'''
{'partial_pos': 264, 'noann_pos': 72}
'''
