import pandas as pd
from tqdm import tqdm
import numpy as np
from cerwsi.utils import read_json_anno


if __name__ == '__main__':
    POSITIVE_CLASS = ['ASC-US', 'LSIL', 'ASC-H', 'HSIL', 'SCC', 'AGC-NOS', 'AGC', 'AGC-N', 'AGC-FN']
    data_root_dir = '/medical-data/data'

    df_jf1 = pd.read_csv('data_resource/cls_pn/group_csv/JFSW_1.csv')
    df_jf2 = pd.read_csv('data_resource/cls_pn/group_csv/JFSW_2.csv')
    df_jf = pd.concat([df_jf1, df_jf2], ignore_index=True)

    ASCUS_pos_ann = []
    for row in tqdm(df_jf.itertuples(index=True), total=len(df_jf), ncols=80):
        if not isinstance(row.json_path, str) or row.kfb_clsname != 'ASC-US':
            continue
        json_path = f'{data_root_dir}/{row.json_path}'
        annos = read_json_anno(json_path)
        pos_ann_num = 0
        for ann in annos:
            sub_class = ann.get('sub_class')
            if sub_class in POSITIVE_CLASS:
                pos_ann_num += 1
        ASCUS_pos_ann.append(pos_ann_num)

    bins = [0, 10, 20, 30, 40, 10000]
    counts, bin_edges = np.histogram(ASCUS_pos_ann, bins=bins)
    print(counts)
    print(f'total ASCUS slide nums: {len(ASCUS_pos_ann)}')

'''
[252  93  21  22  25]
total ASCUS slide nums: 413
'''