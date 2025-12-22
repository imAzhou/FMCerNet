import json

from tqdm import tqdm


RECORD_CLASS = ['NILM', 'GEC', 
                'AGC', 'AGC-N', 'AGC-NOS', 'AGC-FN', 
                'ASC-US','LSIL', 'ASC-H', 'HSIL']

def main():
    for mode in ['train', 'val']:
        anno_lines = []
        with open(f'data_resource/cell_attri/cell_inst/{mode}_cellinst.json', 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        for item in tqdm(json_data, ncols=80):
            clsname = item['sub_class']
            clsid = RECORD_CLASS.index(clsname)
            anno_lines.append(f'{item["filename"]} {clsid}\n')
        with open(f'data_resource/cell_attri/cell_inst/{mode}.txt', 'w') as f:
            f.writelines(anno_lines)
            
            

if __name__ == "__main__":
    main()