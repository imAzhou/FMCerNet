import json
import pandas as pd
from tqdm import tqdm
import os
import shutil

exist_images = [os.path.basename(item) for item in os.listdir('data_resource/0429/224/images/partial_pos')]

for mode in ['train', 'val']:
    with open(f'data_resource/0429/annofiles/partial_{mode}_pos_224.json', 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    new_json_data = []
    pn_cnt = [0,0]
    for patientItem in tqdm(json_data, ncols=80):
        new_patchlist = []
        for patchItem in patientItem['patch_list']:
            if patchItem['filename'] in exist_images:
                new_patchlist.append(patchItem)
                pn_cnt[patchItem['diagnose']] += 1
        patientItem['patch_list'] = new_patchlist
        new_json_data.append(patientItem)
    with open(f'data_resource/0429/annofiles/partial_{mode}_pos_224_filter.json', 'w', encoding='utf-8') as f:
        json.dump(new_json_data, f, ensure_ascii=False, indent=4)
    print(pn_cnt)

# with open('data_resource/0429/annofiles/0422_zheyi_slide_train_224.json', 'r', encoding='utf-8') as f:
#     json_data = json.load(f)
# pn_cnt = [0,0]
# for patientItem in json_data:
#     for patchInfo in patientItem['patchlist']:
#         pn_cnt[patchInfo['diagnose']] += 1
# print(pn_cnt)