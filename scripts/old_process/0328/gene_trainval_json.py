import json
from tqdm import tqdm
import pandas as pd

if __name__ == '__main__':
    
    save_ann_json = 'data_resource/0328/annofiles/zheyi_roi_4fusion_filtered.json'
    with open(save_ann_json,'r') as f:
        total_roiInfo = json.load(f)
    
    csv_root_dir = 'data_resource/slide_anno/0319'
    df_train = pd.read_csv(f'{csv_root_dir}/train.csv')
    train_patientIds = df_train['patientId'].to_list()
    df_val = pd.read_csv(f'{csv_root_dir}/val.csv')
    val_patientIds = df_val['patientId'].to_list()
    
    train_patchs,val_patches = [],[]
    for roiInfo in tqdm(total_roiInfo, ncols=80):
        patientId = roiInfo['patientId']
        if patientId in train_patientIds:
            train_patchs.extend(roiInfo['patchlist'])
        elif patientId in val_patientIds:
            val_patches.extend(roiInfo['patchlist'])
    
    train_path = 'data_resource/0328/annofiles/train4fusion.json'
    with open(train_path,'w') as f:
        json.dump(train_patchs, f)
    val_path = 'data_resource/0328/annofiles/val4fusion.json'
    with open(val_path,'w') as f:
        json.dump(val_patches, f)
        