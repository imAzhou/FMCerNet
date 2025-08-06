import json

WINDOW_SIZE = 850
data_root = f'data_resource/WINDOW_SIZE_{WINDOW_SIZE}'

def main():
    with open(f'{data_root}/ann_jsons/patches_in_RoI_pure_valid.json', 'r', encoding='utf-8') as f:
        RoI_patchlist = json.load(f)
    with open(f'{data_root}/ann_jsons/patches_in_RoI_jfsw_valid.json', 'r', encoding='utf-8') as f:
        jfsw_pos_patchdata = json.load(f)
    
        

if __name__ == "__main__":
    main()