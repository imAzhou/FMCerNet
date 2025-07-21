import json
from datetime import datetime

import pandas as pd
from tqdm import tqdm
import os

invalid_filename = [
    'ZC202110499-CT.kfb',
    '01S462.kfb'
]

def main():
    df_data = pd.read_csv('data_resource/zheyi_annofiles/0607_slide_anno.csv')
    df_abandon = pd.read_csv('data_resource/group_csv/abandon.csv')

    rows_to_add = []
    for row in tqdm(df_data.itertuples(index=False), total=len(df_data), ncols=80):
        filename = os.path.basename(row.kfb_path)
        if filename in invalid_filename:
            row_dict = row._asdict()
            # 重新排列为 df_abandon 的列顺序
            row_ordered = [row_dict[col] for col in df_abandon.columns]
            rows_to_add.append(row_ordered)

    df_abandon = pd.concat([df_abandon, pd.DataFrame(rows_to_add, columns=df_abandon.columns)], ignore_index=True)
    df_abandon.to_csv('data_resource/group_csv/new_abandon.csv', index=False)


if __name__ == "__main__":
    main()

    for item in total_list:
        ann_sorteds = sorted(
            item['annotations'],
            key=lambda d: datetime.strptime(d['updatedTime'], "%Y-%m-%dT%H:%M:%SZ"),
            reverse=True
        )
        annlist = ann_sorteds[0]['annotationResult']
        if len(annlist) < 2:
            continue
        bigroi = 0
        for annitem in annlist:
            if annitem['label'] == 'RoI':
                all_x,all_y = [p[0] for p in annitem['points']], [p[1] for p in annitem['points']]
                x1,x2 = min(all_x),max(all_x)
                y1,y2 = min(all_y),max(all_y)
                w,h = x2-x1, y2-y1
                if w<20 or h<20:
                    continue
                if w<3800 or h<3800:
                    print(f'{item["imageName"]}, small RoI w={w}, h={h}')
                    invalidroi += 1
                elif w>3800 and h>3800:
                    bigroi += 1

