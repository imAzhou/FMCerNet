import json
import pandas as pd
from tqdm import tqdm
import os
from collections import defaultdict

def imgName2patientId()->str:
    name2PID = defaultdict(str)
    for row in tqdm(df_concat.itertuples(index=False), total=len(df_concat), ncols=80):
        filename = os.path.basename(row.kfb_path)
        name2PID[filename] = row.patientId
    return name2PID

def make_jfsw_pos():
    df_zheyi_pos = pd.read_csv('data_resource/0511/zheyi_pos.csv')
    exclude_patientIds = list(df_zheyi_pos['patientId'])
    
    df_slide_year = pd.read_csv('data_resource/group_csv/slide_year.csv')
    df_old = df_slide_year[df_slide_year['year']==0]
    exclude_patientIds.extend(list(df_old['patientId']))
    exclude_patientIds.extend(list(df_abandon['patientId']))
    exclude_patientIds = list(set(exclude_patientIds))

    df_partial_pos = df_concat[
        (df_concat['kfb_clsid'] == 1) &
        (df_concat['anno_path'].apply(lambda x: isinstance(x, str) and x.strip() != '')) &
        (~df_concat['patientId'].isin(exclude_patientIds))
    ]
    df_partial_pos.to_csv(f'{save_dir}/1_jfsw_pos.csv', index=False)

def make_zheyi_pos():
    df_WXL_1 = pd.read_csv(f'data_resource/group_csv/WXL_1.csv')
    with open('data_resource/zheyi_annofiles/宫颈液基细胞—RoI.json', 'r', encoding='utf-8') as f:
        group_json_data = json.load(f)
    with open('data_resource/zheyi_annofiles/宫颈液基细胞—Slide-0409.json', 'r', encoding='utf-8') as f:
        slide0409_json_data = json.load(f)
    with open('data_resource/zheyi_annofiles/宫颈液基细胞—Slide-0422.json', 'r', encoding='utf-8') as f:
        slide0422_json_data = json.load(f)
    
    total_pos_data = []
    
    roi_json_data = []
    for group in group_json_data.values():
        roi_json_data.extend(group)
    new_roi_json_data = []
    for item in roi_json_data:
        patientId = '_'.join(item['imageName'].split('_')[:3])
        abandon_row = df_abandon[df_abandon['patientId'] == patientId]
        if not abandon_row.empty:
            continue
        filtered_df = df_concat[df_concat['patientId'] == patientId].iloc[0]
        row_dict = filtered_df.to_dict()
        row_dict['media_type'] = 'roi'
        row_dict['anno_type'] = 'total'
        total_pos_data.append(row_dict)
        new_roi_json_data.append(item)
    with open('data_resource/zheyi_annofiles/宫颈液基细胞—RoI_filter.json', 'w', encoding='utf-8') as f:
        json.dump(new_roi_json_data, f, ensure_ascii=False)

    name2PID = imgName2patientId()
    for slide_annjson in [slide0409_json_data, slide0422_json_data]:
        for clsname,slidelist in slide_annjson.items():
            for slideinfo in tqdm(slidelist, ncols=80):
                imageName = slideinfo['imageName']
                patientId = name2PID[imageName]
                if patientId != '':
                    filtered_row = df_concat[df_concat['patientId'] == patientId].iloc[0]
                    row_dict = filtered_row.to_dict()
                    row_dict['media_type'] = 'slide'
                    row_dict['anno_type'] = 'total'
                    total_pos_data.append(row_dict)
    
    exist_pids = [i['patientId'] for i in total_pos_data]
    df_filter_WXL_1 = df_WXL_1[
        (df_WXL_1['anno_path'].apply(lambda x: isinstance(x, str) and x.strip() != '')) &
        (~df_WXL_1['patientId'].isin(exist_pids))
    ]
    for row in tqdm(df_filter_WXL_1.itertuples(index=False), total=len(df_filter_WXL_1), ncols=80):
        row_dict = row._asdict()
        row_dict['media_type'] = 'slide'
        row_dict['anno_type'] = 'partial'
        total_pos_data.append(row_dict)

    df_total_pos = pd.DataFrame(total_pos_data)
    df_total_pos.to_csv(f'data_resource/0511/0_zheyi_pos.csv', index=False)

def make_noann_pos():
    df_partial_pos = pd.read_csv(f'{save_dir}/1_jfsw_pos.csv')
    df_total_pos = pd.read_csv(f'{save_dir}/0_zheyi_pos.csv')
    df_ann_pos = pd.concat([df_partial_pos, df_total_pos])

    noann_pos = []
    priority_dict = {
        'ZY_ONLINE_1': 3,
        'WXL_3': 3,
        'WXL_2': 3,
        'WXL_1': 3,
        'JFSW_2': 2,
        'JFSW_1': 2,
    }
    for row in tqdm(df_concat.itertuples(index=False), total=len(df_concat), ncols=80):
        if row.kfb_clsid == 0:
            continue
        annpos_row = df_ann_pos[df_ann_pos['patientId'] == row.patientId]
        if not annpos_row.empty:
            continue
        
        row_dict = row._asdict()
        row_dict['priority'] = priority_dict[row.kfb_source]
        noann_pos.append(row_dict)
    df_noann_pos = pd.DataFrame(noann_pos)
    df_noann_pos.to_csv(f'{save_dir}/2_noann_pos.csv', index=False)

def make_neg():
    neg_slide = []
    priority_dict = {
        'ZY_ONLINE_1': 3,
        'WXL_3': 3,
        'WXL_2': 2,
        'WXL_1': 3,
        'JFSW_2': 1,
        'JFSW_1': 1,
    }
    for row in tqdm(df_concat.itertuples(index=False), total=len(df_concat), ncols=80):
        if row.kfb_clsid == 1:
            continue
        row_dict = row._asdict()
        row_dict['priority'] = priority_dict[row.kfb_source]
        neg_slide.append(row_dict)
    df_neg = pd.DataFrame(neg_slide)
    df_neg.to_csv(f'{save_dir}/3_neg.csv', index=False)

def statistic_pids():
    df_zheyi_pos = pd.read_csv('data_resource/0511/0_zheyi_pos.csv')
    df_jfsw_pos = pd.read_csv('data_resource/0511/1_jfsw_pos.csv')
    df_noann_pos = pd.read_csv('data_resource/0511/2_noann_pos.csv')
    df_neg = pd.read_csv('data_resource/0511/3_neg.csv')

    unique_pids = len(set(df_zheyi_pos['patientId']))
    print(f'0_zheyi_pos: pids {len(df_zheyi_pos["patientId"])}, unique pids {unique_pids}')
    unique_pids = len(set(df_jfsw_pos['patientId']))
    print(f'1_jfsw_pos: pids {len(df_jfsw_pos["patientId"])}, unique pids {unique_pids}')
    unique_pids = len(set(df_noann_pos['patientId']))
    print(f'2_noann_pos: pids {len(df_noann_pos["patientId"])}, unique pids {unique_pids}')
    unique_pids = len(set(df_neg['patientId']))
    print(f'3_neg: pids {len(df_neg["patientId"])}, unique pids {unique_pids}')

if __name__ == "__main__":
    # patientId,kfb_clsname,kfb_clsid,kfb_path,anno_path,kfb_source
    df_abandon = pd.read_csv('data_resource/group_csv/abandon.csv')
    concat_all_data = []
    for kfb_source in ['JFSW_1', 'JFSW_2', 'WXL_1', 'WXL_2', 'WXL_3', 'ZY_ONLINE_1']:
        df_data = pd.read_csv(f'data_resource/group_csv/{kfb_source}.csv')
        concat_all_data.append(df_data)
    df_concat = pd.concat(concat_all_data)
    df_concat = df_concat[~df_concat['patientId'].isin(df_abandon['patientId'])]

    save_dir = 'data_resource/0511'
    os.makedirs(save_dir, exist_ok=True)
    # make_zheyi_pos()
    # make_jfsw_pos()
    # make_noann_pos()
    # make_neg()

    statistic_pids()
    