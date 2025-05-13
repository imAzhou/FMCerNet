import pandas as pd
import random
from prettytable import PrettyTable

NEGATIVE_CLASS = ['NILM', 'GEC']
ASC_CLASS = ['ASC-US', 'LSIL', 'ASC-H', 'HSIL', 'SCC']
AGC_CLASS = ['AGC-NOS', 'AGC', 'AGC-N', 'AGC-FN']

def get_ASCUS(keep_thr = 10):
    df_JFSW_1 = pd.read_csv('data_resource/cls_pn/group_csv/JFSW_1.csv')
    df_JFSW_2 = pd.read_csv('data_resource/cls_pn/group_csv/JFSW_2.csv')
    merged_df = pd.concat([df_JFSW_1, df_JFSW_2], ignore_index=True)

    keep_datas = []
    for row in merged_df.itertuples(index=False):
        cnt,clsname = row.valid_img_cnt, row.kfb_clsname
        if cnt >= keep_thr and cnt < 20 and clsname in ['ASC-US', 'ASC-H']:
            keep_datas.append([row.kfb_path,clsname, row.patientId])
    keep_df = pd.DataFrame(keep_datas, 
                           columns=['kfb_path','kfb_clsname','patientId'])
    return keep_df

def regenerate_csv(filter_ASCUS):
    train_ratio = 0.7
    columns_to_select = ['kfb_path','kfb_clsname','patientId']
    df_all_data = pd.DataFrame()
    for tag in ['JFSW_1', 'JFSW_2', 'WXL_1','WXL_2','WXL_3']:
        file_path = f'data_resource/cls_pn/group_csv/{tag}.csv'
        df = pd.read_csv(file_path, usecols=columns_to_select)
        df['kfb_source'] = tag
        df['kfb_clsid'] = [0 if row.kfb_clsname == 'NILM' else 1 for row in df.itertuples()]
        df_all_data = pd.concat([df_all_data, df], ignore_index=True)

    df_filter_jf = pd.read_csv('data_resource/cls_pn/group_csv/filtered_jfsw_anno.csv', usecols=columns_to_select)
    df_jfsw_1 = pd.read_csv('data_resource/cls_pn/group_csv/JFSW_1.csv', usecols=columns_to_select)
    filtered_data = df_jfsw_1[df_jfsw_1['kfb_clsname'] == 'NILM']
    df_valid_jf = pd.concat([df_filter_jf, filtered_data, filter_ASCUS], ignore_index=True)

    df_train = pd.read_csv('data_resource/cls_pn/train.csv', usecols=columns_to_select)
    df_val = pd.read_csv('data_resource/cls_pn/val.csv', usecols=columns_to_select)
    df_test = pd.read_csv('data_resource/cls_pn/test.csv', usecols=columns_to_select)
    filtered_data = df_test[df_test['kfb_clsname'] == 'NILM']
    df_valid_wxl = pd.concat([df_train, df_val, filtered_data], ignore_index=True)

    df_valid_0 = pd.concat([df_valid_jf, df_valid_wxl], ignore_index=True)
    AGC_path = [row.kfb_path for row in df_valid_0.itertuples() if row.kfb_clsname == 'AGC']
    random.shuffle(AGC_path)
    discard_path = AGC_path[:30]
    temp = [row.kfb_path for row in df_valid_0.itertuples() if row.kfb_clsname in ['AGC-N', 'AGC-NOS']]
    discard_path.extend(temp)

    df_jfsw_2 = pd.read_csv('data_resource/cls_pn/group_csv/JFSW_2.csv', usecols=columns_to_select)
    neg_slide = []
    for row in df_jfsw_2.itertuples(index=False):
        
        if row.kfb_clsname != 'NILM':
            continue
        if (df_valid_0['kfb_path'] == row.kfb_path).any():
            continue
        new_row = df_all_data[df_all_data['kfb_path'] == row.kfb_path].iloc[0]
        neg_slide.append(new_row)
    random.shuffle(neg_slide)
    new_trainval_slide = neg_slide[:265]

    for row in df_valid_0.itertuples(index=False):
        if row.kfb_path not in discard_path:
            new_row = df_all_data[df_all_data['kfb_path'] == row.kfb_path].iloc[0]
            new_trainval_slide.append(new_row)
    
    all_valid_path = [row.kfb_path for row in new_trainval_slide]

    test_df = []
    for row in df_all_data.itertuples(index=False):
        if row.kfb_path not in all_valid_path:
            new_row = df_all_data[df_all_data['kfb_path'] == row.kfb_path].iloc[0]
            test_df.append(new_row)

    total_neg = [row for row in new_trainval_slide if row.kfb_clsname == 'NILM']
    total_pos = [row for row in new_trainval_slide if row.kfb_clsname != 'NILM']
    random.shuffle(total_neg)
    random.shuffle(total_pos)
    train_size = int(len(total_pos)*train_ratio)
    train_df_neg, val_df_neg = total_neg[:train_size],total_neg[train_size:]
    train_df_pos, val_df_pos = total_pos[:train_size],total_pos[train_size:]
    train_df = [*train_df_neg, *train_df_pos]
    val_df = [*val_df_neg, *val_df_pos]
    
    for mode,df_data in zip(['train','val','test'], [train_df, val_df, test_df]):
        slide_clsname_cnt = dict()
        slide_source_cnt = dict()
        new_data = []
        for row in df_data:
            patientId,kfb_clsname,kfb_path = row.patientId,row.kfb_clsname,row.kfb_path
            kfb_clsid,kfb_source = row.kfb_clsid,row.kfb_source
            if mode != 'test':
                kfb_clsname = 'HSIL' if kfb_clsname == 'SCC' else kfb_clsname
            
            new_data.append([patientId, kfb_path, kfb_clsname, kfb_clsid, kfb_source])
            slide_clsname_cnt[kfb_clsname] = slide_clsname_cnt.get(kfb_clsname, 0) + 1
            slide_source_cnt[kfb_source] = slide_source_cnt.get(kfb_source, 0) + 1
        
        new_data_df = pd.DataFrame(new_data, 
                                   columns=['patientId', 'kfb_path', 'kfb_clsname', 'kfb_clsid', 'kfb_source'])
        new_data_df.to_csv(f'data_resource/cls_pn/1127_{mode}.csv', index=False)

        if mode != 'test':
            sorted_keys = ['NILM', 'ASC-US', 'LSIL', 'ASC-H', 'HSIL', 'AGC']
        else:
            sorted_keys = ['NILM', 'ASC-US', 'LSIL', 'ASC-H', 'HSIL', 'SCC', 'AGC','AGC-NOS','AGC-N']
        sorted_values = [slide_clsname_cnt.get(key, 0) for key in sorted_keys]
        result_table = PrettyTable(title=f'{mode} Slide Nums')
        result_table.field_names = ["类别"] + sorted_keys
        result_table.add_row(['num'] + sorted_values)
        print(result_table)

        # result_table = PrettyTable(title=f'{mode} Slide Source')
        # result_table.field_names = ["来源"] + list(slide_source_cnt.keys())
        # result_table.add_row(['num'] + list(slide_source_cnt.values()))
        # print(result_table)
   
    

if __name__ == '__main__':
    filter_ASCUS = get_ASCUS()
    regenerate_csv(filter_ASCUS)
