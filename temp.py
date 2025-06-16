import pandas as pd
df_data = pd.read_csv('data_resource/0511/4_pure_train.csv')
df_neg = df_data[df_data['kfb_clsid']==0]
df_neg.to_csv('data_resource/0511/4_pure_train_negslide.csv', index=False)