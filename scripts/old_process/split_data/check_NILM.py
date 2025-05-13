import pandas as pd
from tqdm import tqdm
from cerwsi.utils import read_json_anno


if __name__ == '__main__':
    POSITIVE_CLASS = ['ASC-US', 'LSIL', 'ASC-H', 'HSIL', 'SCC', 'AGC-NOS', 'AGC', 'AGC-N', 'AGC-FN']
    data_root_dir = '/medical-data/data'

    df_jf1 = pd.read_csv('data_resource/cls_pn/group_csv/JFSW_1.csv')
    df_jf2 = pd.read_csv('data_resource/cls_pn/group_csv/JFSW_2.csv')
    df_jf = pd.concat([df_jf1, df_jf2], ignore_index=True)

    NILM_pos_ann = []
    for row in tqdm(df_jf.itertuples(index=True), total=len(df_jf), ncols=80):
        if not isinstance(row.json_path, str) or row.kfb_clsname != 'NILM':
            continue
        json_path = f'{data_root_dir}/{row.json_path}'
        annos = read_json_anno(json_path)
        
        for ann in annos:
            sub_class = ann.get('sub_class')
            if sub_class in POSITIVE_CLASS:
                NILM_pos_ann.append(f'{row.kfb_path}\n')
                break
    with open('data_resource/cls_pn/group_csv/jfsw_NILM_pos_ann.txt', 'w') as f:
        f.writelines(NILM_pos_ann)
        