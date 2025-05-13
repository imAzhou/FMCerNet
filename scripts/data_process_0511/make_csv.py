import json
import pandas as pd
from tqdm import tqdm
import os

def imgName2patientId()->str:
    name2PID = {}
    for row in tqdm(df_concat.itertuples(index=False), total=len(df_concat), ncols=80):
        filename = os.path.basename(row.kfb_path)
        name2PID[filename] = row.patientId
    return name2PID

def make_partial_pos():
    df_0409 = pd.read_csv('data_resource/zheyi_annofiles/0409_slide_anno.csv')
    df_0422 = pd.read_csv('data_resource/zheyi_annofiles/0422_slide_anno.csv')
    df_total_posslide = pd.concat([df_0409, df_0422])
    exclude_patientIds = list(df_total_posslide['patientId'])
    with open('data_resource/zheyi_annofiles/宫颈液基细胞—RoI_filter.json', 'r', encoding='utf-8') as f:
        roi_json_data = json.load(f)
    for item in roi_json_data:
        patientId = '_'.join(item['imageName'].split('_')[:3])
        if len(item['annotations'][0]['annotationResult']) > 0:
            exclude_patientIds.append(patientId)
    df_slide_year = pd.read_csv('data_resource/slide_anno/group_csv/slide_year.csv')
    df_old = df_slide_year[df_slide_year['year']==0]
    exclude_patientIds.extend(list(df_old['patientId']))
    exclude_patientIds.extend(list(df_abandon['patientId']))
    exclude_patientIds = list(set(exclude_patientIds))

    df_partial_pos = df_concat[
        (df_concat['kfb_clsid'] == 1) &
        (df_concat['anno_path'].apply(lambda x: isinstance(x, str) and x.strip() != '')) &
        (~df_concat['patientId'].isin(exclude_patientIds))
    ]
    df_partial_pos.to_csv(f'{save_dir}/partial_pos.csv', index=False)

def make_total_pos():
    with open('data_resource/zheyi_annofiles/宫颈液基细胞—RoI.json', 'r', encoding='utf-8') as f:
        group_json_data = json.load(f)
    with open('data_resource/zheyi_annofiles/宫颈液基细胞—Slide-0409.json', 'r', encoding='utf-8') as f:
        slide0409_json_data = json.load(f)
    with open('data_resource/zheyi_annofiles/宫颈液基细胞—Slide-0422.json', 'r', encoding='utf-8') as f:
        slide0422_json_data = json.load(f)
    roi_json_data = []
    for group in group_json_data.values():
        roi_json_data.extend(group)
    total_pos_data = []
    new_roi_json_data = []
    for item in roi_json_data:
        patientId = '_'.join(item['imageName'].split('_')[:3])
        abandon_row = df_abandon[df_abandon['patientId'] == patientId]
        if not abandon_row.empty:
            continue
        filtered_df = df_concat[df_concat['patientId'] == patientId].iloc[0]
        row_dict = filtered_df.to_dict()
        row_dict['media_type'] = 'roi'
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
                abandon_row = df_abandon[df_abandon['patientId'] == patientId]
                if abandon_row.empty:
                    filtered_row = df_concat[df_concat['patientId'] == patientId].iloc[0]
                    row_dict = filtered_row.to_dict()
                    row_dict['media_type'] = 'slide'
                    total_pos_data.append(row_dict)
    df_total_pos = pd.DataFrame(total_pos_data)
    df_total_pos.to_csv(f'data_resource/slide_anno/0511/total_pos.csv', index=False)

def make_noann_pos():
    df_partial_pos = pd.read_csv(f'{save_dir}/partial_pos.csv')
    df_total_pos = pd.read_csv(f'{save_dir}/total_pos.csv')
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
        abandon_row = df_abandon[df_abandon['patientId'] == row.patientId]
        if not abandon_row.empty:
            continue
        annpos_row = df_ann_pos[df_ann_pos['patientId'] == row.patientId]
        if not annpos_row.empty:
            continue
        
        row_dict = row._asdict()
        row_dict['priority'] = priority_dict[row.kfb_source]
        noann_pos.append(row_dict)
    df_noann_pos = pd.DataFrame(noann_pos)
    df_noann_pos.to_csv(f'{save_dir}/noann_pos.csv', index=False)

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
    df_neg.to_csv(f'{save_dir}/neg.csv', index=False)


if __name__ == "__main__":
    # patientId,kfb_clsname,kfb_clsid,kfb_path,anno_path,kfb_source
    df_abandon = pd.read_csv('data_resource/slide_anno/group_csv/abandon.csv')
    concat_all_data = []
    for kfb_source in ['JFSW_1', 'JFSW_2', 'WXL_1', 'WXL_2', 'WXL_3', 'ZY_ONLINE_1']:
        df_data = pd.read_csv(f'data_resource/slide_anno/group_csv/{kfb_source}.csv')
        concat_all_data.append(df_data)
    df_concat = pd.concat(concat_all_data)

    save_dir = 'data_resource/slide_anno/0511'
    os.makedirs(save_dir, exist_ok=True)
    make_total_pos()
    make_partial_pos()
    make_noann_pos()
    make_neg()
    