import json
from tqdm import tqdm

WINDOW_SIZE = 1600
data_root = f'data_resource/0630/WINDOW_SIZE_{WINDOW_SIZE}'

def main():
    with open(f'{data_root}/ann_jsons/patches_in_RoI_pure_valid.json', 'r', encoding='utf-8') as f:
        RoI_patchlist = json.load(f)
    with open(f'{data_root}/ann_jsons/patches_in_RoI_jfsw_valid.json', 'r', encoding='utf-8') as f:
        jfsw_pos_patchdata = json.load(f)
    purename2pid = {}
    for datalist in [RoI_patchlist, jfsw_pos_patchdata]:
        for item in tqdm(datalist, ncols=80):
            purename = item['filename'].split('.')[0]
            purename2pid[purename] = item['patientId']
    
    with open(f'{data_root}/annofiles/purename2pId.json', 'w', encoding='utf-8') as f:
        json.dump(purename2pid, f, ensure_ascii=False)

if __name__ == "__main__":
    main()