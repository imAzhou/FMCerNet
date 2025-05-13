import pandas as pd
from tqdm import tqdm

df_train = pd.read_csv('data_resource/0416/annofiles/train_0422.csv')
df_val = pd.read_csv('data_resource/0416/annofiles/val_0422.csv')
df_anno = pd.read_csv('data_resource/zheyi_annofiles/0422_slide_anno.csv')

save_dir = 'data_resource/0429_2/annofiles'
new_rows = []
for row in tqdm(df_anno.itertuples(index=False), total=len(df_anno), ncols=80):
    # 判断是否有值为 'value' 的行
    mask_train = df_train['patientId'] == row.patientId
    mask_val = df_val['patientId'] == row.patientId
    if mask_train.any() or mask_val.any():
        continue
    new_rows.append(row)

df_train = pd.concat([df_train, pd.DataFrame(new_rows)], ignore_index=True)
df_train.to_csv(f'{save_dir}/train.csv', index=False)
df_val.to_csv(f'{save_dir}/val.csv', index=False)