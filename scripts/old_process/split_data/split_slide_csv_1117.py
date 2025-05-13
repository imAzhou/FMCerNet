import pandas as pd
import os
import glob
import random
from pathlib import Path

test_kfbs = [
    ['/disk/medical_datasets/cervix/ZJU-TCT/第一批标注2023.9.5之前/NILM/01S028.kfb', 0, '01S028'],
    # ['/disk/medical_datasets/cervix/ZJU-TCT/第二批标注2023.9.5/NILM/01S404.kfb', 0, '01S404'],
    # ['/disk/medical_datasets/cervix/ZJU-TCT/第二批标注2023.9.5/NILM/01S349.kfb', 0, '01S349'],
    # ['/disk/medical_datasets/cervix/ZJU-TCT/第一批标注2023.9.5之前/NILM/01S042.kfb', 0, '01S042'],
    # ['/disk/medical_datasets/cervix/ZJU-TCT/第一批标注2023.9.5之前/HSIL/01S175.kfb', 1, '01S175'],
    # ['/disk/medical_datasets/cervix/ZJU-TCT/第一批标注2023.9.5之前/LSIL/01S023.kfb', 1, '01S023'],
    # ['/disk/medical_datasets/cervix/ZJU-TCT/第二批标注2023.9.5/AGC-N/01S260.kfb', 1, '01S260'],
]
# df_data = pd.DataFrame(test_kfbs, columns=['kfb_path', 'kfb_clsid', 'patientId'])
# df_data.to_csv('data_resource/test_kfb_file.csv', index=False)


# for mode in ['train','val', 'test']:
#     csv_file = f'data_resource/cls_pn/{mode}.csv'
#     df_csv = pd.read_csv(csv_file)
#     new_data = []
#     for row in df_csv.itertuples(index=False):
#         patientId = os.path.basename(row.kfb_path).split('.')[0]
#         kfb_clsname = row.kfb_path.split('/')[-2]
#         new_data.append([row.kfb_path, row.kfb_clsid, kfb_clsname, patientId])
#     df_data = pd.DataFrame(new_data, columns=['kfb_path', 'kfb_clsid', 'kfb_clsname', 'patientId'])
#     df_data.to_csv(f'data_resource/cls_pn/new_{mode}.csv', index=False)


# JFSW_root_dir = '/disk/medical_datasets/cervix/JFSW'
# positive_kfb = glob.glob(f'{JFSW_root_dir}/阳性/*.kfb')
# negative_kfb = glob.glob(f'{JFSW_root_dir}/阴性/*.kfb')
# positive_diag = '/disk/medical_datasets/cervix/JFSW/JFSW病人整片分类.xlsx'

# new_data = []

# df_positive_diag = pd.read_excel(positive_diag, sheet_name='阳性')
# column_list = df_positive_diag.columns.tolist()
# selected_columns = df_positive_diag[df_positive_diag.columns.tolist()]
# for index, row in selected_columns.iterrows():
#     patientId = f'JFSW_1_{index}'
#     kfb_clsid = 1
#     filename, kfb_clsname = row['name'], row['diagnostic_type']
#     kfb_path = f'{JFSW_root_dir}/阳性/{filename}'
#     assert kfb_path in positive_kfb, f'kfb_path {kfb_path} none exist.'
#     new_data.append([kfb_path, kfb_clsid, kfb_clsname, patientId])

# new_data.extend([[kfb_path, 0, 'NILM', f'JFSW_1_{idx+len(positive_kfb)}'] for idx,kfb_path in enumerate(negative_kfb)])

# df_data = pd.DataFrame(new_data, columns=['kfb_path', 'kfb_clsid', 'kfb_clsname', 'patientId'])
# df_data.to_csv(f'data_resource/cls_pn/test_jfsw01.csv', index=False)


# root_dir = '/disk/medical_datasets/cervix/negative_WSI'
# kfb_list = glob.glob(f'{root_dir}/**/*.kfb')
# new_data = [[kfb_path, 0, 'NILM', f'total_Neg_{idx}'] for idx, kfb_path in enumerate(kfb_list)]
# df_data = pd.DataFrame(new_data, columns=['kfb_path', 'kfb_clsid', 'kfb_clsname', 'patientId'])
# df_data.to_csv(f'data_resource/cls_pn/test_total_neg.csv', index=False)


# csv_file = '/disk/medical_datasets/cervix/positive_WSI/slide_info.csv'
# df_csv = pd.read_csv(csv_file)
# new_data = []
# for row in df_csv.itertuples(index=False):
#     patientId = 'WXL-P-' + os.path.basename(row.kfb_path).split('.')[0]
#     kfb_clsname = row.diag_type
#     new_data.append([row.kfb_path, row.kfb_clsid, kfb_clsname, patientId])
# df_data = pd.DataFrame(new_data, columns=['kfb_path', 'kfb_clsid', 'kfb_clsname', 'patientId'])
# df_data.to_csv(f'data_resource/cls_pn/test_wxl_p.csv', index=False)

# root_dir = '/disk/medical_datasets/cervix/JFSW_1109'
# kfb_list = glob.glob(f'{root_dir}/**/*.kfb')
# new_data = []
# for idx,kfb_path in enumerate(kfb_list):
#     patientId = f'JFSW_2_{idx}'
#     kfb_clsname = kfb_path.split('/')[-2]
#     kfb_clsid = 0 if kfb_clsname == 'NILM' else 1
#     new_data.append([kfb_path, kfb_clsid, kfb_clsname, patientId])
# df_data = pd.DataFrame(new_data, columns=['kfb_path', 'kfb_clsid', 'kfb_clsname', 'patientId'])
# df_data.to_csv(f'data_resource/cls_pn/test_jfsw02.csv', index=False)

# csv_files = [
#     'data_resource/cls_pn/test.csv',
#     'data_resource/cls_pn/test_jfsw01.csv',
#     'data_resource/cls_pn/test_jfsw02.csv',
#     'data_resource/cls_pn/test_total_neg.csv',
#     'data_resource/cls_pn/test_wxl_p.csv'
# ]
# dataframes = [pd.read_csv(file) for file in csv_files]
# merged_df = pd.concat(dataframes, ignore_index=True)
# merged_df.to_csv('test_all.csv', index=False)

def merge_trainval():
    total_data = []
    train_ratio = 0.7

    # JF-1: 30阳，JF-2：375阳 18 阴
    df_jfsw_anno = pd.read_csv('data_resource/cls_pn/jfsw_valid_ann/filtered_jfsw_anno.csv')
    exist_NILM = []
    for row in df_jfsw_anno.itertuples(index=False):
        kfb_path, kfb_clsname, patientId = row.kfb_path, row.kfb_clsname, row.patientId
        kfb_clsid = 1
        if kfb_clsname == 'NILM':
            kfb_clsid = 0
            exist_NILM.append(kfb_path)
        kfb_source = 'JFSW_1' if 'JFSW_1' in patientId else 'JFSW_2'
        total_data.append([kfb_path, kfb_clsid, kfb_clsname, patientId, kfb_source])

    # JF-2: 180 阴 随机选择
    all_kfb_paths = glob.glob('/disk/medical_datasets/cervix/JFSW_1109/NILM/*.kfb')
    temp_kfb_paths = [kfb_path for kfb_path in all_kfb_paths if kfb_path not in exist_NILM]
    random_neg = random.sample(temp_kfb_paths, 180)
    for idx, kfb_path in enumerate(random_neg):
        total_data.append([kfb_path, 0, 'NILM', f'JFSW_2_N{idx}', 'JFSW_2'])

    # JF-1: 100 阴
    all_kfb_paths = glob.glob('/disk/medical_datasets/cervix/JFSW/阴性/*.kfb')
    for idx, kfb_path in enumerate(all_kfb_paths):
        total_data.append([kfb_path, 0, 'NILM', f'JFSW_1_N{idx}', 'JFSW_1'])

    # WXL-1: 41 阳 41 阴 + 107
    df_train = pd.read_csv('data_resource/cls_pn/train.csv')
    df_val = pd.read_csv('data_resource/cls_pn/val.csv')
    df_test = pd.read_csv('data_resource/cls_pn/test.csv')
    idx = 0
    for mode_i, df_data in enumerate([df_train, df_val, df_test]):
        for row in df_data.itertuples(index=False):
            kfb_path, kfb_clsname, kfb_clsid = row.kfb_path, row.kfb_clsname, row.kfb_clsid
            if mode_i == 2 and kfb_clsname != 'NILM':   # test.csv 中只保留阴性 slide
                continue
            patientId = f'WXL_1_N{idx}'
            total_data.append([kfb_path, kfb_clsid, kfb_clsname, patientId, 'WXL_1'])
            idx += 1
    
    random.shuffle(total_data)
    total_posi = [item for item in total_data if item[1] == 1]
    total_nega = [item for item in total_data if item[1] == 0]

    train_slide_num = int(len(total_posi)*train_ratio)
    train_posi = total_posi[:train_slide_num]
    train_nega = total_nega[:train_slide_num]
    train_data = [*train_posi, *train_nega]

    val_posi = total_posi[train_slide_num:]
    val_nega = total_nega[train_slide_num:]
    val_data = [*val_posi, *val_nega]

    df_train_data = pd.DataFrame(train_data, columns=['kfb_path', 'kfb_clsid', 'kfb_clsname', 'patientId', 'kfb_source'])
    df_val_data = pd.DataFrame(val_data, columns=['kfb_path', 'kfb_clsid', 'kfb_clsname', 'patientId', 'kfb_source'])
    df_train_data.to_csv('data_resource/cls_pn/1117_train.csv', index=False)
    df_val_data.to_csv('data_resource/cls_pn/1117_val.csv', index=False)

def merge_test():
    total_data = []

    # JF-1: 30阳，JF-2：375阳 18 阴
    df_jfsw_anno = pd.read_csv('data_resource/cls_pn/jfsw_valid_ann/filtered_jfsw_anno.csv')
    df_positive_diag = pd.read_excel('/disk/medical_datasets/cervix/JFSW/JFSW病人整片分类.xlsx', sheet_name='阳性')
    exist_positive = []
    for row in df_jfsw_anno.itertuples(index=False):
        kfb_path, kfb_clsname, patientId = row.kfb_path, row.kfb_clsname, row.patientId
        if kfb_clsname != 'NILM' and 'JFSW_1' in patientId:
            exist_positive.append(kfb_path)
    idx = 0
    for kfb_path in glob.glob('/disk/medical_datasets/cervix/JFSW/阳性/*.kfb'):
        if kfb_path not in exist_positive:
            filename = os.path.basename(kfb_path)
            filtered_rows = df_positive_diag[df_positive_diag['new_name'] == filename]
            assert len(filtered_rows) == 1
            kfb_clsname = filtered_rows.iloc[0]['diagnostic_type']
            patientId = f'JFSW_1_P{idx}'
            total_data.append([kfb_path, 1, kfb_clsname, patientId, 'JFSW_1'])
            idx += 1
    idx = 0
    df_test = pd.read_csv('data_resource/cls_pn/test.csv')
    for row in df_test.itertuples(index=False):
        kfb_path, kfb_clsname, kfb_clsid = row.kfb_path, row.kfb_clsname, row.kfb_clsid
        if kfb_clsname != 'NILM':   # test.csv 中只保留阳性 slide
            patientId = f'WXL_1_P{idx}'
            total_data.append([kfb_path, kfb_clsid, kfb_clsname, patientId, 'WXL_1'])
            idx += 1
    idx = 0
    csv_files = [
        'data_resource/cls_pn/1117_train.csv',
        'data_resource/cls_pn/1117_val.csv'
    ]
    dataframes = [pd.read_csv(file) for file in csv_files]
    df_trainval = pd.concat(dataframes, ignore_index=True)
    exist_kfbs = df_trainval['kfb_path'].tolist()
    all_jfsw_2_kfbs = glob.glob('/disk/medical_datasets/cervix/JFSW_1109/**/*.kfb')
    for kfb_path in all_jfsw_2_kfbs:
        if kfb_path in exist_kfbs:
            continue
        path = Path(kfb_path)
        kfb_clsname = path.parts[-2]
        patientId = f'JFSW_2_{idx}'
        kfb_clsid = 0 if kfb_clsname == 'NILM' else 1
        total_data.append([kfb_path, kfb_clsid, kfb_clsname, patientId, 'JFSW_2'])
        idx += 1
    
    all_kfb_path = glob.glob('/disk/medical_datasets/cervix/negative_WSI/**/*.kfb')
    for idx,kfb_path in enumerate(all_kfb_path):
        total_data.append([kfb_path, 0, 'NILM', f'WXL_2_{idx}', 'WXL_2'])

    idx = 0
    df_wxl_3 = pd.read_csv('/disk/medical_datasets/cervix/positive_WSI/slide_info.csv')
    for row in df_wxl_3.itertuples(index=False):
        kfb_path, kfb_clsname, kfb_clsid = row.kfb_path, row.diag_type, row.kfb_clsid
        total_data.append([kfb_path, kfb_clsid, kfb_clsname, f'WXL_3_{idx}', 'WXL_3'])
        idx += 1

    df_test_data = pd.DataFrame(total_data, columns=['kfb_path', 'kfb_clsid', 'kfb_clsname', 'patientId', 'kfb_source'])
    df_test_data.to_csv('data_resource/cls_pn/1117_test.csv', index=False)

if __name__ == '__main__':
    merge_trainval()
    merge_test()