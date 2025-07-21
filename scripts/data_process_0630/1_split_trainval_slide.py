import pandas as pd
import random

from tqdm import tqdm

def main():
    df_zheyi_pos = pd.read_csv('data_resource/0630/0_zheyi_pos.csv')
    df_jfsw_pos = pd.read_csv('data_resource/0630/1_jfsw_pos.csv')
    df_noann_pos = pd.read_csv('data_resource/0630/2_noann_pos.csv')
    df_neg = pd.read_csv('data_resource/0630/3_neg.csv')
    df_sorted = df_neg.sort_values(by='priority', ascending=False)

    set_totalpos_pure = list(set(df_zheyi_pos['patientId']))
    df_topNeg = df_sorted.iloc[0:1500]
    pure_trainval_patientIds = [*set_totalpos_pure, *list(df_topNeg['patientId'])]   # 2994: 1500 neg+1494 pos
    
    df_agc_sample = df_jfsw_pos[df_jfsw_pos['kfb_clsname'] == 'AGC'].sample(n=300, random_state=42)
    df_jfsw_keep = df_jfsw_pos.drop(df_agc_sample.index)
    jfsw_train_patientIds = [
        *list(set(df_jfsw_keep['patientId'])),    # 312 pos
        *df_sorted.iloc[1500:1900]['patientId'].to_list()    # 400 neg
    ]   # 712: 400 neg+312 pos
    drop_jfsw_pids = ['JFSW_2_2293', 'JFSW_2_2276', 'JFSW_2_2147', 'JFSW_2_2288']
    jfsw_train_patientIds = [i for i in jfsw_train_patientIds if i not in drop_jfsw_pids]
    
    test_patientIds = [
        *df_sorted.iloc[1900:]['patientId'].to_list(),   # 3770 neg
        *list(df_noann_pos['patientId']),    # 694 pos
        *list(df_agc_sample['patientId']),    # 300 pos
    ]   # 4764

    list_of_dfs = [df_zheyi_pos, df_jfsw_pos, df_noann_pos, df_neg]
    common_cols = list(set.intersection(*[set(df.columns) for df in list_of_dfs]))
    df_concat = pd.concat([df[common_cols] for df in list_of_dfs], ignore_index=True)
    tags = ['4_pure_trainval','5_jfsw_train','7_test']
    for idx,pid_groups in enumerate([pure_trainval_patientIds, jfsw_train_patientIds, test_patientIds]):
        data_rows = []
        for pid in tqdm(pid_groups, ncols=80):
            filtered_df = df_concat[df_concat['patientId'] == pid].iloc[0]
            data_rows.append(filtered_df)
        df_data_rows = pd.DataFrame(data_rows)
        df_data_rows.to_csv(f'data_resource/0630/{tags[idx]}.csv', index=False)

def eval_spilt():
    '''
    从 patientId 看，
    pure_train 和 jfsw_train 的 patientId 不重复
    val 和 test 中不应该出现 pure_train 和 jfsw_train 的 patientId
    '''
    df_jfsw_train = pd.read_csv('data_resource/0630/5_jfsw_train.csv')
    df_pure_train = pd.read_csv('data_resource/0630/4_pure_train.csv')
    df_val = pd.read_csv('data_resource/0630/6_val.csv')
    df_test = pd.read_csv('data_resource/0630/7_test.csv')

    pid_jfsw_train = list(df_jfsw_train['patientId'])
    error_flag = False
    for pid in list(df_pure_train['patientId']):
        if pid in pid_jfsw_train:
            error_flag = True
            print(f'ERROR: pure_train {pid} in jfsw_train patientIds.')
    if not error_flag:
        print('pure_train is clear of jfsw_train.')

    pid_fusion_train = [*list(df_pure_train['patientId']), *list(df_jfsw_train['patientId'])]
    error_flag = False
    for pid in list(df_val['patientId']):
        if pid in pid_fusion_train:
            error_flag = True
            print(f'ERROR: val {pid} in fusion_train patientIds.')
    if not error_flag:
        print('val is clear of patientId.')
    
    error_flag = False
    for pid in list(df_test['patientId']):
        if pid in pid_fusion_train:
            error_flag = True
            print(f'ERROR: test {pid} in fusion_train patientIds.')
    if not error_flag:
        print('test is clear of patientId.')


if __name__ == "__main__":
    main()
    # eval_spilt()
