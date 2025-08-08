import json
import os
from tqdm import tqdm
import random
import pickle
from collections import defaultdict


def gene_multilabel(pred_result, old_train_data, purename2pId, save_path):
    thr = 0.5
    sample_step = 10
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
                    'gt_label': []
                })  
    for patchlist in patientErrorlist.values():
        random.shuffle(patchlist)
        samplelist = patchlist[:sample_step]
        old_train_data['data_list'].extend(samplelist)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(old_train_data, f, ensure_ascii=False)

if __name__ == '__main__':
    log_dir = 'log/WS1600/mlc/hardsample_round0'
    hardsample_jsondir = 'data_resource/WINDOW_SIZE_1600/hardsample_annofiles'
    old_train_jsonpath = f'{hardsample_jsondir}/multilable_hs_round0.json'
    new_train_savepath = f'{hardsample_jsondir}/multilable_hs_round1.json'
    purename2pId_jsonpath = 'data_resource/WINDOW_SIZE_1600/annofiles/purename2pId.json'
    
    with open(f"{log_dir}/pred_result.pkl", "rb") as f:
        pred_result = pickle.load(f)
    with open(old_train_jsonpath, 'r', encoding='utf-8') as f:
        old_train_data = json.load(f)
    with open(purename2pId_jsonpath, 'r', encoding='utf-8') as f:
        purename2pId = json.load(f)
    
    gene_multilabel(pred_result, old_train_data, purename2pId, new_train_savepath)

    pn_cnt = [0,0]
    with open(new_train_savepath, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    for item in tqdm(json_data['data_list'], ncols=80):
        diagnose = int(len(item['gt_label'])>0)
        pn_cnt[diagnose] += 1
    print(pn_cnt)
        
