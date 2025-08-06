import json
from tqdm import tqdm
from pycocotools.coco import COCO

WINDOW_SIZE = 400

def main():
    root_dir = 'data_resource/ComparisonDetectorDataset'
    json_save_dir = f'{root_dir}/WINDOW_SIZE_{WINDOW_SIZE}/annofiles'
    for mode in ['train','val']:
        pn_cnt = [0,0]
        with open(f'{json_save_dir}/{mode}.json', 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        coco = COCO(f'{json_save_dir}/{mode}.json')

        multilabel_jsondata = {
            "metainfo": {
                "classes":[i['name'] for i in json_data["categories"]]
            },
            "data_list": []
        }
        binarylabel_txtdata = []

        for imgitem in tqdm(json_data['images'], ncols=80):
            annids = coco.getAnnIds([imgitem['id']])
            annos = coco.loadAnns(annids)

            multilabel_jsondata['data_list'].append({
                "img_path": imgitem["file_name"],
                "gt_label": list(set([ann['category_id']-1 for ann in annos]))
            })
            binarylabel_txtdata.append(f'{imgitem["file_name"]} {imgitem["diagnose"]}\n')
            pn_cnt[imgitem['diagnose']] += 1
        print(f'{mode} pn_cnt: {pn_cnt}')

        with open(f'{json_save_dir}/multilabel_{mode}.json', 'w', encoding='utf-8') as f:
            json.dump(multilabel_jsondata, f, ensure_ascii=False)
        with open(f'{json_save_dir}/binarylabel_{mode}.txt', 'w', encoding='utf-8') as f:
            f.writelines(binarylabel_txtdata)

if __name__ == "__main__":
    main()

'''
WINDOW_SIZE = 400
train pn_cnt: [27489, 22524]
val pn_cnt: [3367, 2503]
'''