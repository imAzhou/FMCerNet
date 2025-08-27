import json
import os
from tqdm import tqdm
import random
import pickle
from collections import defaultdict


def gene_from_traindata(pred_result, old_train_data, purename2pId):
    thr = 0.5
    sample_step = 15
    exist_purename = [os.path.basename(
        item['img_path']).split('.')[0] for item in old_train_data['data_list']]
    patientErrorlist = defaultdict(list)
    for item in tqdm(pred_result, ncols=80):
        gt_diagnose = int(len(item.gt_label)>0)
        pred_diagnose = int(item.img_prob > 0.5)
        pred_multi_label = [clsidx for clsidx,cls_score in enumerate(item.pos_prob) if cls_score > thr]
        if gt_diagnose == 0 and (pred_diagnose == 1 or len(pred_multi_label)>1):
            purename = os.path.basename(item.img_path).split('.')[0]
            if purename not in exist_purename:
                patientErrorlist[purename2pId[purename]].append({
                    'img_path': f'neg/{purename}.png',
                    'gt_label': [],
                })  
    data_list = []
    for patchlist in patientErrorlist.values():
        random.shuffle(patchlist)
        samplelist = patchlist[:sample_step]
        data_list.extend(samplelist)
    print(f'Add new {len(data_list)} neg patches from train data.')
    return data_list

def gene_from_negslide(hsIn_negslide_jsonpath):
    with open(hsIn_negslide_jsonpath, 'r', encoding='utf-8') as f:
        hsIn_negslide = json.load(f)
    data_list = []
    for pinfo in hsIn_negslide:
        if pinfo["filename"] != 'ZY_ONLINE_1_3422_round1_4.png':
            data_list.append({
                'img_path': f'{pinfo["prefix"]}/{pinfo["filename"]}',
                'gt_label': [],
            })
    print(f'Add new {len(data_list)} neg patches from neg slide.')
    return data_list

if __name__ == '__main__':
    # log_dir = 'log/WS850/wscernet/2025_08_25_11_15_43'
    log_dir = 'log/WS850/mlc/hs_round0'
    hardsample_jsondir = 'data_resource/WINDOW_SIZE_850/hardsample_annofiles'
    old_train_jsonpath = f'{hardsample_jsondir}/multilable_hs_round0.json'
    new_train_savepath = f'{hardsample_jsondir}/multilable_hs_round1.json'
    hsIn_negslide_jsonpath = f'{hardsample_jsondir}/patches_in_negslide_hs1.json'
    purename2pId_jsonpath = 'data_resource/WINDOW_SIZE_850/annofiles/purename2pId.json'

    with open(f"{log_dir}/pred_result.pkl", "rb") as f:
        pred_result = pickle.load(f)
    with open(purename2pId_jsonpath, 'r', encoding='utf-8') as f:
        purename2pId = json.load(f)
    with open(old_train_jsonpath, 'r', encoding='utf-8') as f:
        old_train_data = json.load(f)
    
    traindata_patchlist = gene_from_traindata(pred_result, old_train_data, purename2pId)
    old_train_data['data_list'].extend(traindata_patchlist)
    negslide_patchlist = gene_from_negslide(hsIn_negslide_jsonpath)
    old_train_data['data_list'].extend(negslide_patchlist)
    with open(new_train_savepath, 'w', encoding='utf-8') as f:
        json.dump(old_train_data, f, ensure_ascii=False)

    for jsonpath in [old_train_jsonpath, new_train_savepath]:
        pn_cnt = [0,0]
        with open(jsonpath, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        for item in tqdm(json_data['data_list'], ncols=80):
            diagnose = int(len(item['gt_label'])>0)
            pn_cnt[diagnose] += 1
        print(pn_cnt)


'''
hs_round0: [11468, 12607]
Add new 2775 neg patches from train data.
Add new 4871 neg patches from neg slide.
hs_round1: [19114, 12607]
Add new 7768 neg patches from train data.
Add new 4295 neg patches from neg slide.
hs_round2: [28402, 12607]
'''