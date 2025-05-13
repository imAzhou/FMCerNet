import pandas as pd
from tqdm import tqdm
import random
import os
from prettytable import PrettyTable

def load_pos_noanno():
    with open('/medical-data/data/cervix/JFSW_1109/anno_miss.txt', 'r') as f:
        lines = f.readlines()
        lines = [l.strip().replace('/disk/medical_datasets/','') for l in lines]
    return lines

def load_neg_withpos():
    with open('/medical-data/data/cervix/JFSW_1109/NILM_with_pos.txt', 'r') as f:
        lines = f.readlines()
        lines = [l.strip() for l in lines]
    return lines

def split_trainvaltest():
    df_jf1 = pd.read_csv('data_resource/slide_anno/group_csv/JFSW_1.csv')
    df_jf2 = pd.read_csv('data_resource/slide_anno/group_csv/JFSW_2.csv')
    df_wxl1 = pd.read_csv('data_resource/slide_anno/group_csv/WXL_1.csv')
    df_wxl2 = pd.read_csv('data_resource/slide_anno/group_csv/WXL_2.csv')
    df_wxl3 = pd.read_csv('data_resource/slide_anno/group_csv/WXL_3.csv')
    df_zyonline_1 = pd.read_csv('data_resource/slide_anno/group_csv/ZY_ONLINE_1.csv')

    train,val,test = [],[],[]

    for row in tqdm(df_jf1.itertuples(index=False), total=len(df_jf1), desc='Process JF_1'):
        kfb_clsid = 0 if row.kfb_clsname == 'NILM' else 1
        anno_path = row.json_path
        new_row = [row.patientId, row.kfb_clsname, kfb_clsid, row.kfb_path, anno_path, 'JFSW_1']
        if random.random() < 0.8:
            train.append(new_row)
        else:
            val.append(new_row)
    
    noanno_lines = load_pos_noanno()
    neg_pass_lines = load_neg_withpos()
    for row in tqdm(df_jf2.itertuples(index=False), total=len(df_jf2), desc='Process JF_2'):
        kfb_clsid = 0 if row.kfb_clsname == 'NILM' else 1
        anno_path = row.json_path
        new_row = [row.patientId, row.kfb_clsname, kfb_clsid, row.kfb_path, anno_path, 'JFSW_2']
        if row.kfb_path in noanno_lines:
            test.append(new_row)
            continue
        if row.kfb_path in neg_pass_lines:
            continue
        if random.random() < 0.8:
            train.append(new_row)
        else:
            val.append(new_row)
    
    for row in tqdm(df_wxl1.itertuples(index=False), total=len(df_wxl1), desc='Process WXL_1'):
        anno_path = row.kfb_path.replace('.kfb','.xml')
        new_row = [row.patientId, row.kfb_clsname, row.kfb_clsid, row.kfb_path, anno_path, 'WXL_1']
        if os.path.exists(f'/medical-data/data/{anno_path}'):
            if random.random() < 0.8:
                train.append(new_row)
            else:
                val.append(new_row)
        else:
            new_row[-2] = ''
            if row.kfb_clsname == 'NILM':
                if random.random() < 0.8:
                    train.append(new_row)
                else:
                    val.append(new_row)
            else:
                test.append(new_row)
    
    for row in tqdm(df_wxl2.itertuples(index=False), total=len(df_wxl2), desc='Process WXL_2'):
        new_row = [row.patientId, row.kfb_clsname, row.kfb_clsid, row.kfb_path, '', 'WXL_2']
        test.append(new_row)
    
    for row in tqdm(df_wxl3.itertuples(index=False), total=len(df_wxl3), desc='Process WXL_3'):
        new_row = [row.patientId, row.kfb_clsname, row.kfb_clsid, row.kfb_path, '', 'WXL_3']
        if row.kfb_clsname == 'NILM':
            if random.random() < 0.8:
                train.append(new_row)
            else:
                val.append(new_row)
        else:
            test.append(new_row)
    
    for row in tqdm(df_zyonline_1.itertuples(index=False), total=len(df_zyonline_1), desc='Process ZY_ONLINE_1'):
        new_row = [row.patientId, row.kfb_clsname, row.kfb_clsid, row.kfb_path, '', 'ZY_ONLINE_1']
        if row.kfb_clsname != 'NILM':
            test.append(new_row)
        elif random.random() < 0.5:
            test.append(new_row)
        else:
            if random.random() < 0.8:
                train.append(new_row)
            else:
                val.append(new_row)

    columns = ['patientId', 'kfb_clsname', 'kfb_clsid', 'kfb_path', 'anno_path', 'kfb_source']
    df_train = pd.DataFrame(train, columns=columns)
    df_val = pd.DataFrame(val, columns=columns)
    df_test = pd.DataFrame(test, columns=columns)

    for mode,df_data in zip(['train','val','test'],[df_train,df_val,df_test]):
        cls_counts = df_data['kfb_clsname'].value_counts()
        total_count = cls_counts.sum()
        table = PrettyTable(title=f'{mode} Mode')
        table.field_names = list(cls_counts.index) + ["Total"]
        row_values = list(cls_counts.values) + [total_count]
        table.add_row(row_values)
        print(table)

        df_data.to_csv(f'data_resource/slide_anno/0319/{mode}.csv', index=False)

def split_test():
    df_test = pd.read_csv('data_resource/slide_anno/0319/test.csv')
    pos,neg = [],[]
    for row in df_test.itertuples(index=False):
        if row.kfb_clsid == 1:
            pos.append(row)
        else:
            neg.append(row)
    
    selected_idxs = random.sample(range(len(neg)), len(pos))
    pos_neg_data = [*pos, *[neg[i] for i in selected_idxs]]
    neg_data = [neg[i] for i in range(len(neg)) if i not in selected_idxs]

    df_test_posneg = pd.DataFrame(pos_neg_data, columns=df_test.columns)
    df_test_posneg.to_csv('data_resource/slide_anno/0319/test_posneg.csv', index=False)
    df_test_neg = pd.DataFrame(neg_data, columns=df_test.columns)
    df_test_neg.to_csv('data_resource/slide_anno/0319/test_neg.csv', index=False)

if __name__ == '__main__':
    # split_trainvaltest()
    split_test()

'''
+---------------------------------------------------------------------------+
|                                 train Mode                                |
+------+------+-----+------+--------+-------+-------+---------+-----+-------+
| NILM | LSIL | AGC | HSIL | ASC-US | ASC-H | AGC-N | AGC-NOS | SCC | Total |
+------+------+-----+------+--------+-------+-------+---------+-----+-------+
| 2074 | 442  | 381 | 348  |  338   |  310  |   3   |    1    |  1  |  3898 |
+------+------+-----+------+--------+-------+-------+---------+-----+-------+
+-----------------------------------------------------------+
|                          val Mode                         |
+------+------+------+-----+-------+--------+-------+-------+
| NILM | LSIL | HSIL | AGC | ASC-H | ASC-US | AGC-N | Total |
+------+------+------+-----+-------+--------+-------+-------+
| 531  |  99  |  93  |  88 |   88  |   79   |   1   |  979  |
+------+------+------+-----+-------+--------+-------+-------+
+---------------------------------------------------------------------------+
|                                 test Mode                                 |
+------+--------+------+------+-------+-----+-------+-----+---------+-------+
| NILM | ASC-US | LSIL | HSIL | ASC-H | SCC | AGC-N | AGC | AGC-NOS | Total |
+------+--------+------+------+-------+-----+-------+-----+---------+-------+
| 3204 |  478   | 239  |  76  |   64  |  16 |   6   |  4  |    4    |  4091 |
+------+--------+------+------+-------+-----+-------+-----+---------+-------+
'''