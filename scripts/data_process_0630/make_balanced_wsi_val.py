'''
WSI Val: neg 300+1000=1300 pos 310+994=1304
0907日:
训练集：AGC+150, ASC-US+280, NILM+400
'''

import pandas as pd


def main():
    df_train = pd.read_csv('data_resource/0630/WINDOW_SIZE_1600/annofiles/45_purejfsw_train.csv')
    df_val = pd.read_csv('data_resource/0630/WINDOW_SIZE_1600/annofiles/67_wsi_val.csv')
    df_add_neg = df_val[df_val['kfb_clsid']==0].sample(n=400, random_state=42)
    df_add_agc = df_val[df_val['kfb_clsname']=='AGC'].sample(n=150, random_state=42)
    df_add_ascus = df_val[df_val['kfb_clsname']=='ASC-US'].sample(n=280, random_state=42)

    df_new_train = pd.concat([df_train, df_add_neg, df_add_agc, df_add_ascus])
    df_new_train.to_csv('data_resource/0630/WINDOW_SIZE_1600/annofiles/45_0907_train.csv', index=False)
    cls_stats = df_new_train['kfb_clsname'].value_counts().reset_index()
    cls_stats.columns = ['kfb_clsname', 'count']
    print(cls_stats)

    df_to_remove = pd.concat([df_add_neg, df_add_agc, df_add_ascus])
    df_new_val = df_val.drop(df_to_remove.index)
    df_new_val.to_csv('data_resource/0630/WINDOW_SIZE_1600/annofiles/67_0907_val.csv', index=False)
    cls_stats = df_new_val['kfb_clsname'].value_counts().reset_index()
    cls_stats.columns = ['kfb_clsname', 'count']
    print(cls_stats)


    


if __name__ == "__main__":
    main()