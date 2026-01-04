import json
from tqdm import tqdm
from collections import Counter, defaultdict


def main():
    jsonpath = f'data_resource/cell_attri/cell_inst/filter_train_cellinst.json'
    with open(jsonpath, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    attrv_list = defaultdict(list)
    for cellitem in tqdm(train_data, ncols=80):
        attr_v = ','.join([str(i) for i in cellitem['attr_v']])
        attrv_list[attr_v].append(cellitem)
    for attv,celllist in attrv_list.items():
        if len(celllist) == 1:
            print(f"clsname: {celllist[0]['sub_class']}, desc: {celllist[0]['jfsw_desc']}")
    

if __name__ == "__main__":
    main()