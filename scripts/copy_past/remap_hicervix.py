import json
import pandas as pd
from prettytable import PrettyTable
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
import random

dataroot = 'data_resource/HiCervix'
json_savepath = 'scripts/copy_past/hicervix.json'
clsname_map = {
    'Normal': 'NILM',
    'ECC': 'NILM',
    'RPC': 'NILM',
    'TRI': 'NILM',
    'MPC': 'NILM',
    'PG': 'NILM',
    'EMC': 'NILM',
    'Atrophy': 'NILM',
    'HSV': 'NILM',
    'CC': 'NILM',
    'HCG': 'NILM',
    'FUNGI': 'NILM',
    'ACTINO': 'NILM',
    'AGC-NOS': 'AGC',
    'AGC': 'AGC',
    'ADC': 'AGC',
    'AGC-FN': 'AGC',
    'AGC-ECC-NOS': 'AGC',
    'AGC-EMC-NOS': 'AGC',
    'ADC-ECC': 'AGC',
    'ADC-EMC': 'AGC',
    'ASC-US': 'ASC-US',
    'ASC-H': 'ASC-H',
    'LSIL': 'LSIL',
    'HSIL': 'HSIL',
    'SCC': 'HSIL',
}

def statistic():
    total_df = []
    total_w, total_h = [],[]
    class_counts = defaultdict(int)
    save_patchlist = []
    for mode in ['train','val','test']:
        df_data = pd.read_csv(f'{dataroot}/{mode}.csv')
        total_df.append(df_data)
        for row in tqdm(df_data.itertuples(index=False), total=len(df_data), ncols=80):
            clsname = clsname_map[row.class_name]
            if clsname != 'NILM':
                imgpath = f'{dataroot}/{mode}/{row.image_name}'
                img = Image.open(imgpath)
                w,h = img.size
                if w<1600 and h<1600:
                    if clsname == 'AGC' and random.random() > 0.25:
                        continue
                    total_w.append(w)
                    total_h.append(h)
                    class_counts[clsname] += 1
                    save_patchlist.append({
                        'clsname': clsname,
                        'imgpath': imgpath,
                        'wh': (w,h),
                        'area': w*h
                    })
    with open(json_savepath, 'w', encoding='utf-8') as f:
        json.dump(save_patchlist, f, ensure_ascii=False)                
            
    # total_df = pd.concat(total_df)
    # total_df["mapped_class"] = total_df["class_name"].map(clsname_map)
    # class_counts = total_df["mapped_class"].value_counts().to_dict()
    # total_samples = len(total_df)
    table = PrettyTable()
    table.field_names = ["Class Name", "Count"]
    poscls_cnt = 0
    for cls, count in class_counts.items():
        table.add_row([cls, count])
        if cls != 'NILM':
            poscls_cnt += count
    print(table)
    print(f'Total pos cls cnt: {poscls_cnt}')
    print(f'total width: min({min(total_w)}), max({max(total_w)})')
    print(f'total height: min({min(total_h)}), max({max(total_h)})')
    

if __name__ == "__main__":
    statistic()


'''
+---------------------+-------+------------+
| Class Name (Mapped) | Count | Percentage |
+---------------------+-------+------------+
|         NILM        | 23000 |   57.17%   |
|         AGC         |  8389 |   20.85%   |
|         HSIL        |  3060 |   7.61%    |
|        ASC-US       |  2599 |   6.46%    |
|         LSIL        |  1744 |   4.34%    |
|        ASC-H        |  1437 |   3.57%    |
+---------------------+-------+------------+
Total pos cls cnt: 17229
total width: min(58), max(3826)
total height: min(63), max(2160)

+---------------------+-------+
| Class Name (Mapped) | Count |
+---------------------+-------+
|         LSIL        |  1589 |
|        ASC-H        |  1417 |
|         HSIL        |  2774 |
|        ASC-US       |  2507 |
|         AGC         |  2015 |
+---------------------+-------+
Total pos cls cnt: 10302
total width: min(58), max(1599)
total height: min(63), max(1591)
'''