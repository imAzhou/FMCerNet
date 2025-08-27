import pandas as pd


df_data = pd.read_csv('data_resource/0630/WINDOW_SIZE_1600/annofiles/45_purejfsw_train.csv')
df_data = df_data.drop_duplicates(subset=["patientId"])
print(len(df_data[df_data['kfb_clsid']==0]))
print(len(df_data[df_data['kfb_clsid']==1]))
