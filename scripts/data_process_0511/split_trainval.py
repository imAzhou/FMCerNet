import pandas as pd
import random

from tqdm import tqdm

def main():
    df_total_pos = pd.read_csv('data_resource/0511/total_pos.csv')
    df_partial_pos = pd.read_csv('data_resource/0511/partial_pos.csv')
    df_noann_pos = pd.read_csv('data_resource/0511/noann_pos.csv')
    df_neg = pd.read_csv('data_resource/0511/neg.csv')
    list_of_dfs = [df_total_pos, df_partial_pos, df_noann_pos, df_neg]
    common_cols = list(set.intersection(*[set(df.columns) for df in list_of_dfs]))
    df_concat = pd.concat([df[common_cols] for df in list_of_dfs], ignore_index=True)

    df_totalpos_pure = df_total_pos[~df_total_pos['patientId'].isin(df_partial_pos['patientId'])]
    set_totalpos_pure = list(set(df_totalpos_pure['patientId']))

    random.shuffle(set_totalpos_pure)
    val_pos_patientIds = set_totalpos_pure[:300]
    df_top300 = df_neg.sort_values(by='priority', ascending=False).head(300)
    val_patientIds = [*val_pos_patientIds, *list(df_top300['patientId'])]   # 600: 300 neg+300 pos

    totalpos_train = df_total_pos[~df_total_pos['patientId'].isin(val_pos_patientIds)]
    df_sorted = df_neg.sort_values(by='priority', ascending=False)
    totalneg_train = df_sorted.iloc[301:1201]
    pure_train_patientIds = [*list(set(totalpos_train['patientId'])), *list(totalneg_train['patientId'])]   # 1632: 900 neg+732 pos
    
    df_rest_pos = df_partial_pos[~df_partial_pos['patientId'].isin(df_total_pos['patientId'])]
    train_with_partial = [
        *pure_train_patientIds,    # 900 neg+732 pos
        *list(set(df_rest_pos['patientId'])),    # 938 pos
        *df_sorted.iloc[1201:2201]['patientId'].to_list()    # 1000 neg
    ]   # 3570: 1900 neg+1670 pos
    
    test_patientIds = [
        *df_sorted.iloc[2201:]['patientId'].to_list(),   # 3633 neg
        *list(df_noann_pos['patientId']),    # 1177 pos
    ]

    tags = ['pure_train','fusion_train','val','test']
    for idx,pid_groups in enumerate([pure_train_patientIds, train_with_partial, val_patientIds, test_patientIds]):
        data_rows = []
        for pid in tqdm(pid_groups, ncols=80):
            filtered_df = df_concat[df_concat['patientId'] == pid].iloc[0]
            data_rows.append(filtered_df)
        df_data_rows = pd.DataFrame(data_rows)
        df_data_rows.to_csv(f'data_resource/0511/{tags[idx]}.csv', index=False)


if __name__ == "__main__":
    main()