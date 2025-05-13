import json
from tqdm import tqdm
import numpy as np
from prettytable import PrettyTable

POSITIVE_CLASS = ['AGC', 'ASC-US','LSIL', 'ASC-H', 'HSIL']

if __name__ == '__main__':
    root_dir = 'data_resource/0328/annofiles'
    total_clsname_cnt = []
    for mode in ['train', 'val']:
        clsname_cnt = [0]*len(POSITIVE_CLASS)

        json_path = f'{root_dir}/{mode}_patches_v0328.json'
        with open(json_path, 'r') as f:
            anno_data = json.load(f)
        for ann in tqdm(anno_data, ncols=80):
            if ann['diagnose'] == 0:
                continue
            for clsname in ann['clsnames']:
                clsname_cnt[POSITIVE_CLASS.index(clsname)] += 1
        total_clsname_cnt.append(clsname_cnt)
    total_clsname_cnt = np.array(total_clsname_cnt)
    result_table = PrettyTable()
    result_table.field_names = ['Mode'] + POSITIVE_CLASS
    result_table.add_row(['Train'] + total_clsname_cnt[0].tolist())
    result_table.add_row(['Val'] + total_clsname_cnt[1].tolist())
    result_table.add_row(['Total'] + np.sum(total_clsname_cnt, axis=0).tolist())
    print(result_table)
