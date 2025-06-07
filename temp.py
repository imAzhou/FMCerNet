import glob
import json
import shutil
import os
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

df_data = pd.read_csv('data_resource/0511/2_noann_pos.csv')
df_data = df_data[~df_data['kfb_source'].isin(['JFSW_1', 'JFSW_2'])]
print(len(df_data))
counts = df_data['kfb_clsname'].value_counts()
print(counts)