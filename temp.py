
import json


valid_patchlist_jsonpath = '/medical-data_NB/data/cervix/slide_patches/valid_patches_WS1600.json'

def main():
    with open(valid_patchlist_jsonpath, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

if __name__ == "__main__":
    main()