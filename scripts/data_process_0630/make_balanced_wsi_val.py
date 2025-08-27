'''
WSI Val: neg 300+1000=1300 pos 310+994=1304
'''

import pandas as pd


def main():
    df_val = pd.read_csv('data_resource/0630/6_val.csv')
    df_test = pd.read_csv('data_resource/0630/7_test.csv')
    df_test_neg = df_test[df_test['kfb_clsid']==0].sample(n=1000, random_state=42)
    df_test_pos = df_test[df_test['kfb_clsid']==1]
    df_data = pd.concat([df_val, df_test_neg, df_test_pos])
    df_data.to_csv('data_resource/0630/WINDOW_SIZE_1600/annofiles/67_wsi_val.csv', index=False)
    print()

if __name__ == "__main__":
    main()