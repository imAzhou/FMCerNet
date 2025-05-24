import pandas as pd
import random

from tqdm import tqdm

def main():
    df_zheyi_pos = pd.read_csv('data_resource/0511/0_zheyi_pos.csv')
    df_jfsw_pos = pd.read_csv('data_resource/0511/1_jfsw_pos.csv')
    df_noann_pos = pd.read_csv('data_resource/0511/2_noann_pos.csv')
    df_neg = pd.read_csv('data_resource/0511/3_neg.csv')

    df_totalpos_pure = df_zheyi_pos[df_zheyi_pos['anno_type']=='total']
    set_totalpos_pure = list(set(df_totalpos_pure['patientId']))

    random.shuffle(set_totalpos_pure)
    val_pos_patientIds = set_totalpos_pure[:300]
    df_top300 = df_neg.sort_values(by='priority', ascending=False).head(300)
    val_patientIds = [*val_pos_patientIds, *list(df_top300['patientId'])]   # 600: 300 neg+300 pos

    totalpos_train = df_zheyi_pos[~df_zheyi_pos['patientId'].isin(val_pos_patientIds)]
    df_sorted = df_neg.sort_values(by='priority', ascending=False)
    totalneg_train = df_sorted.iloc[301:1201]
    pure_train_patientIds = [*list(set(totalpos_train['patientId'])), *list(totalneg_train['patientId'])]   # 1624: 900 neg+724 pos
    
    jfsw_train_patientIds = [
        *list(set(df_jfsw_pos['patientId'])),    # 876 pos
        *df_sorted.iloc[1201:2201]['patientId'].to_list()    # 1000 neg
    ]   # 1876: 1000 neg+876 pos
    
    test_patientIds = [
        *df_sorted.iloc[2201:]['patientId'].to_list(),   # 3469 neg
        *list(df_noann_pos['patientId']),    # 902 pos
    ]

    list_of_dfs = [df_zheyi_pos, df_jfsw_pos, df_noann_pos, df_neg]
    common_cols = list(set.intersection(*[set(df.columns) for df in list_of_dfs]))
    df_concat = pd.concat([df[common_cols] for df in list_of_dfs], ignore_index=True)
    tags = ['4_pure_train','5_jfsw_train','6_val','7_test']
    for idx,pid_groups in enumerate([pure_train_patientIds, jfsw_train_patientIds, val_patientIds, test_patientIds]):
        data_rows = []
        for pid in tqdm(pid_groups, ncols=80):
            filtered_df = df_concat[df_concat['patientId'] == pid].iloc[0]
            data_rows.append(filtered_df)
        df_data_rows = pd.DataFrame(data_rows)
        df_data_rows.to_csv(f'data_resource/0511/{tags[idx]}.csv', index=False)

def eval_spilt():
    '''
    从 patientId 看，
    pure_train 和 jfsw_train 的 patientId 不重复
    val 和 test 中不应该出现 pure_train 和 jfsw_train 的 patientId
    '''
    df_jfsw_train = pd.read_csv('data_resource/0511/5_jfsw_train.csv')
    df_pure_train = pd.read_csv('data_resource/0511/4_pure_train.csv')
    df_val = pd.read_csv('data_resource/0511/6_val.csv')
    df_test = pd.read_csv('data_resource/0511/7_test.csv')

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
    eval_spilt()
    # df_data = pd.read_csv('data_resource/0511/7_test.csv')
    # pids = len(df_data[df_data['kfb_clsid'] == 0])
    # print(pids)