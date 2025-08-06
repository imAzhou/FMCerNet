import json
<<<<<<< HEAD
from tqdm import tqdm

WINDOW_SIZE = 1600
data_root = f'data_resource/0630/WINDOW_SIZE_{WINDOW_SIZE}'
=======

WINDOW_SIZE = 850
data_root = f'data_resource/WINDOW_SIZE_{WINDOW_SIZE}'
>>>>>>> e14f4888d1e4228c257149865d6deb152971c162

def main():
    with open(f'{data_root}/ann_jsons/patches_in_RoI_pure_valid.json', 'r', encoding='utf-8') as f:
        RoI_patchlist = json.load(f)
    with open(f'{data_root}/ann_jsons/patches_in_RoI_jfsw_valid.json', 'r', encoding='utf-8') as f:
        jfsw_pos_patchdata = json.load(f)
<<<<<<< HEAD
    purename2pid = {}
    for datalist in [RoI_patchlist, jfsw_pos_patchdata]:
        for item in tqdm(datalist, ncols=80):
            purename = item['filename'].split('.')[0]
            purename2pid[purename] = item['patientId']
    
    with open(f'{data_root}/annofiles/purename2pId.json', 'w', encoding='utf-8') as f:
        json.dump(purename2pid, f, ensure_ascii=False)
=======
    
        
>>>>>>> e14f4888d1e4228c257149865d6deb152971c162

if __name__ == "__main__":
    main()