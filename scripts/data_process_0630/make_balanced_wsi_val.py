'''
WSI Val: neg 300+1000=1300 pos 310+994=1304
0907日:
训练集：AGC+150, ASC-US+280, NILM+400
'''

import pandas as pd
from prettytable import PrettyTable

def main():
    df_puretrain = pd.read_csv('data_resource/0630/4_pure_train.csv')
    df_jfswtrain = pd.read_csv('data_resource/0630/5_jfsw_train.csv')
    df_val = pd.read_csv('data_resource/0630/6_val.csv')
    df_test = pd.read_csv('data_resource/0630/7_test.csv')
    
    df_add_neg = df_test[df_test['kfb_clsid']==0].sample(n=400, random_state=42)
    df_add_agc = df_test[df_test['kfb_clsname']=='AGC'].sample(n=150, random_state=42)
    df_add_ascus = df_test[df_test['kfb_clsname']=='ASC-US'].sample(n=280, random_state=42)

    df_new_train = pd.concat([df_puretrain, df_jfswtrain, df_add_neg, df_add_agc, df_add_ascus])
    df_new_train.to_csv('data_resource/0630/45_0924_train.csv', index=False)
    cls_stats = df_new_train['kfb_clsname'].value_counts().reset_index()
    cls_stats.columns = ['kfb_clsname', 'count']
    print(cls_stats)

    df_to_remove = pd.concat([df_add_neg, df_add_agc, df_add_ascus])
    df_new_test = df_test.drop(df_to_remove.index)
    df_add_neg_val = df_new_test[df_new_test['kfb_clsid']==0].sample(n=600, random_state=42)
    df_add_pos_val = df_new_test[df_new_test['kfb_clsid']==1]
    df_new_val = pd.concat([df_val, df_add_neg_val, df_add_pos_val])
    df_new_val.to_csv('data_resource/0630/67_0924_val.csv', index=False)
    cls_stats = df_new_val['kfb_clsname'].value_counts().reset_index()
    cls_stats.columns = ['kfb_clsname', 'count']
    print(cls_stats)

def show_existcsv():
    df_puretrain = pd.read_csv('data_resource/0630/4_pure_train.csv')
    df_jfswtrain = pd.read_csv('data_resource/0630/5_jfsw_train.csv')
    df_val = pd.read_csv('data_resource/0630/6_val.csv')
    df_test = pd.read_csv('data_resource/0630/7_test.csv')
    dfs = {
        "pure_train": df_puretrain,
        "jfsw_train": df_jfswtrain,
        "val": df_val,
        "test": df_test
    }
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
    }

    all_classes = ['NILM','AGC','ASC-US','LSIL','ASC-H','HSIL']

    # PrettyTable 初始化
    table = PrettyTable()
    table.field_names = ["kfb_clsname"] + list(dfs.keys())

    for cls in all_classes:
        row = [cls]
        for name, df in dfs.items():
            # 先做映射
            mapped_cls = df['kfb_clsname'].map(RECORD_CLASS)
            count = (mapped_cls == cls).sum()
            row.append(count)
        table.add_row(row)

    print(table)
    


if __name__ == "__main__":
    # show_existcsv()
    main()