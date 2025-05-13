import json
from tqdm import tqdm
import pandas as pd
import numpy as np
import shutil

def filter_zheyi_roi():
    save_ann_json = 'data_resource/0328/annofiles/zheyi_roi.json'
    with open(save_ann_json,'r') as f:
        total_roiInfo = json.load(f)
    total_pids = []
    for roiInfo in tqdm(total_roiInfo, ncols=80):
        patientId = roiInfo['patientId']
        total_pids.append(patientId)
    return total_pids

if __name__ == '__main__':
    csv_root_dir = 'data_resource/slide_anno/0319'
    df_train = pd.read_csv(f'{csv_root_dir}/train.csv')
    train_patientIds = df_train['patientId'].to_list()
    df_val = pd.read_csv(f'{csv_root_dir}/val.csv')
    val_patientIds = df_val['patientId'].to_list()
    
    invalid_pids = [
        'JFSW_1_57', 'JSFW_1_67', 'JSFW_2_125', 'JSFW_2_1323', 'JSFW_2_1327', 'JSFW_2_1583', 'JSFW_2_1521',
        'JFSW_2_837',
        'JFSW_2_1308', 'JFSW_2_255', 'JFSW_2_360',
    ]
    excluded_patientIds = filter_zheyi_roi()

    jfsw_json_dir = 'data_resource/0403/annofiles'
    for mode in ['train', 'val']:
        original_pn_cnt = [0,0]
        zheyiroi_pn_cnt = [0,0]
        zheyislide_pn_cnt = [0,0]
        with open(f'{jfsw_json_dir}/{mode}_patches_v0403.json','r') as f:
            original_anno = json.load(f)
        with open(f'data_resource/0328/annofiles/{mode}4fusion.json','r') as f:
            zheyiroi_anno = json.load(f)
        with open('data_resource/0328/annofiles/zheyi_slide_4fusion.json','r') as f:
            zheyislide_anno = json.load(f)

        total_imginfo = []
        for imginfo in tqdm(original_anno, ncols=80):
            filename = imginfo['filename']
            patientId = '_'.join(filename.split('_')[:3])
            if patientId in [*invalid_pids, *excluded_patientIds]:
                continue
            del imginfo['clsid']
            prefix = imginfo["prefix"]
            imginfo['prefix'] = '0403jfsw/images/' + imginfo['prefix']
            total_imginfo.append(imginfo)
            original_pn_cnt[imginfo['diagnose']] += 1
            if imginfo['diagnose'] == 1:
                src_path = f'data_resource/0403/images/{prefix}'
            else:
                src_path = f'data_resource/0319/images/{prefix}'
            shutil.move(
                f'{src_path}/{imginfo["filename"]}',
                f'data_resource/0410/{imginfo["prefix"]}/{imginfo["filename"]}'
            )
        
        for imginfo in tqdm(zheyiroi_anno, ncols=80):
            prefix = imginfo["prefix"]
            imginfo['prefix'] = '0328roi/images/' + imginfo['prefix']
            total_imginfo.append(imginfo)
            zheyiroi_pn_cnt[imginfo['diagnose']] += 1
            src_path = f'data_resource/0328/images4fusion/{prefix}'
            shutil.move(
                f'{src_path}/{imginfo["filename"]}',
                f'data_resource/0410/{imginfo["prefix"]}/{imginfo["filename"]}'
            )
        
        if mode == 'train':
            zheyislide_patchlist = []
            for slideinfo in tqdm(zheyislide_anno, ncols=80):
                zheyislide_patchlist.extend(slideinfo['patchlist'])
            for imginfo in tqdm(zheyislide_patchlist, ncols=80):
                prefix = 'Pos' if imginfo['diagnose'] == 1 else 'Neg'
                imginfo['prefix'] = '0410slide/images/' + prefix
                total_imginfo.append(imginfo)
                zheyislide_pn_cnt[imginfo['diagnose']] += 1
                src_path = f'data_resource/0328/0410slide/{prefix}'
                shutil.move(
                    f'{src_path}/{imginfo["filename"]}',
                    f'data_resource/0410/{imginfo["prefix"]}/{imginfo["filename"]}'
                )
        
        with open(f'data_resource/0328/annofiles/{mode}_v0410.json','w') as f:
            json.dump(total_imginfo, f)

        print(original_pn_cnt)
        print(zheyiroi_pn_cnt)
        print(zheyislide_pn_cnt)
        print(np.sum([original_pn_cnt, zheyiroi_pn_cnt, zheyislide_pn_cnt], axis=0))

'''
Train: 
original_pn_cnt: [45305, 25086]
zheyiroi_pn_cnt: [5256, 7057]
zheyislide_pn_cnt: [1744, 1330]
Total: [52305 33473]

Val:
original_pn_cnt: [11551, 7187]
zheyiroi_pn_cnt: [1430, 1774]
Total: [12981  8961]
'''