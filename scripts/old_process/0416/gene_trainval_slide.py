import pandas as pd
import os

from prettytable import PrettyTable
from tqdm import tqdm

classes = ['NILM', 'AGC', 'ASC-US', 'LSIL', 'ASC-H', 'HSIL' ]
RECORD_CLASS = {
    'NILM': 'NILM',
    'ASC-US': 'ASC-US',
    'LSIL': 'LSIL',
    'ASC-H': 'ASC-H',
    'HSIL': 'HSIL',
    'SCC': 'HSIL',
    'AGC-N': 'AGC',
    'AGC': 'AGC',
    'AGC-NOS': 'AGC',
    'AGC-FN': 'AGC',
}

def main():
    invalid_pids = [
        'JFSW_1_57', 'JSFW_1_67', 'JSFW_2_125', 'JSFW_2_1323', 'JSFW_2_1327', 'JSFW_2_1583', 'JSFW_2_1521',
        'JFSW_2_837',
        'JFSW_2_1308', 'JFSW_2_255', 'JFSW_2_360',
    ]
    
    old_dir = 'data_resource/slide_anno/0319'
    new_dir = 'data_resource/0416/annofiles'
    os.makedirs(new_dir, exist_ok=True)

    for mode in ['train','val']:
        df_data = pd.read_csv(f'{old_dir}/{mode}.csv')
        filtered_df = df_data[~df_data['patientId'].isin(invalid_pids)]
        
        table = PrettyTable(title=f'{mode} Mode')
        table.field_names = classes + ["Pos", "Total"]
        cls_cnt = [0]*len(classes)
        cls_counts = filtered_df['kfb_clsname'].value_counts()
        for cls_name, count in cls_counts.items():
            cls_cnt[classes.index(RECORD_CLASS[cls_name])] += count
        
        row_values = cls_cnt + [sum(cls_cnt[1:]), sum(cls_cnt)]
        table.add_row(row_values)
        print(table)

        filtered_df.to_csv(f'{new_dir}/{mode}.csv', index=False)


if __name__ == "__main__":
    main()

'''
+----------------------------------------------------------+
|                        train Mode                        |
+------+-----+--------+------+-------+------+------+-------+
| NILM | AGC | ASC-US | LSIL | ASC-H | HSIL | Pos  | Total |
+------+-----+--------+------+-------+------+------+-------+
| 2074 | 385 |  343   | 448  |  313  | 351  | 1840 |  3914 |
+------+-----+--------+------+-------+------+------+-------+
+---------------------------------------------------------+
|                         val Mode                        |
+------+-----+--------+------+-------+------+-----+-------+
| NILM | AGC | ASC-US | LSIL | ASC-H | HSIL | Pos | Total |
+------+-----+--------+------+-------+------+-----+-------+
| 531  |  89 |   79   |  99  |   88  |  93  | 448 |  979  |
+------+-----+--------+------+-------+------+-----+-------+
'''