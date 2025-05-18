import json
import matplotlib.pyplot as plt
from collections import Counter
import os

from tqdm import tqdm


# result_savedir = f'statistic_results/0511/dataset_analyze/{tag}'
# os.makedirs(result_savedir, exist_ok=True, mode=0o777)


def pn_analyze():
    for tag in ['puretrain', 'fusiontrain', 'val']:
        with open(f'data_resource/0511/annofiles/{tag}_coco.json') as f:
            coco_data = json.load(f)

        pn_cnt = [0,0]
        for img in tqdm(coco_data['images'], ncols=80):
            pn_cnt[img['diagnose']] += 1
        print(f'{tag}: {pn_cnt}')

if __name__ == "__main__":
    pn_analyze()

'''
puretrain: [98154, 38570]
fusiontrain: [142150, 103616]
val: [38189, 9919]
'''