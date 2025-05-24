import os
import pandas as pd

df_abandon = pd.read_csv('data_resource/group_csv/abandon_add.csv')
concat_all_data = []
for kfb_source in ['JFSW_1', 'JFSW_2', 'WXL_1', 'WXL_2', 'WXL_3', 'ZY_ONLINE_1']:
    df_data = pd.read_csv(f'data_resource/group_csv/{kfb_source}.csv')
    concat_all_data.append(df_data)
df_concat = pd.concat(concat_all_data)
df_concat = df_concat[~df_concat['patientId'].isin(df_abandon['patientId'])]

# 提取每个kfb_path的文件名作为新列
df_concat['filename'] = df_concat['kfb_path'].apply(os.path.basename)

# 找出有重复文件名的行
duplicate_files = df_concat[df_concat.duplicated('filename', keep=False)]

# 按照文件名分组，每个组包含相同文件名的所有行
grouped = duplicate_files.groupby('filename')

# 遍历每个分组并处理
str_group = []
for filename, group in grouped:
    str_group.append(f"\n文件名: {filename}")
    str_group.append(f"共找到 {len(group)} 条记录:\n")
    for row in group.itertuples():
        str_group.append(f'{row.patientId}: {row.kfb_clsname}, {row.kfb_path}\n')
    str_group.append('\n')

with open('data_resource/group_csv/duplicated_left.txt', 'w') as f:
    f.writelines(str_group)

# add_abandon = []
# df_abandon = pd.read_csv('data_resource/group_csv/abandon.csv')
# for filename, group in grouped:
#     for row in group.itertuples(index=False):
#         if row.kfb_source == 'WXL_3' or row.kfb_source == 'JFSW_1':
#             abandon_row = df_abandon[df_abandon['patientId'] == row.patientId]
#             if abandon_row.empty:
#                 filtered_row = df_concat[df_concat['patientId'] == row.patientId].iloc[0]
#                 filtered_row = filtered_row.drop('filename')
#                 add_abandon.append(filtered_row)
# df_add_abandon = pd.DataFrame(add_abandon)
# df_concat_abandon = pd.concat([df_abandon, df_add_abandon])
# df_concat_abandon.to_csv('data_resource/group_csv/abandon_add.csv', index=False)
