import json
from tqdm import tqdm
from collections import defaultdict
from prettytable import PrettyTable
from pycocotools.coco import COCO

WINDOW_SIZE = 400
root_dir = 'data_resource/ComparisonDetectorDataset'
json_save_dir = f'{root_dir}/WINDOW_SIZE_{WINDOW_SIZE}/annofiles'

def statistic():
    for tag in ['train','test']:
        with open(f'{json_save_dir}/multilabel_{tag}.json', 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        classes = json_data["metainfo"]["classes"]
        data_list = json_data["data_list"]

        # 每个类别 -> 出现过的 tile 数
        label_count = defaultdict(int)
        for item in data_list:
            gt_labels = item.get("gt_label", [])
            for label in gt_labels:
                label_count[label] += 1

        table = PrettyTable(title=tag)
        table.field_names = classes
        table.add_row([label_count.get(i, 0) for i in range(len(classes))])
        print(table)

def main():
    for mode in ['train','test']:
        pn_cnt = [0,0]
        with open(f'{json_save_dir}/{mode}.json', 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        coco = COCO(f'{json_save_dir}/{mode}.json')

        multilabel_jsondata = {
            "metainfo": {
                "classes":[i['name'] for i in json_data["categories"][:-1]]
            },
            "data_list": []
        }

        for imgitem in tqdm(json_data['images'], ncols=80):
            annids = coco.getAnnIds([imgitem['id']])
            annos = coco.loadAnns(annids)
            gt_label = []
            for ann in annos:
                if ann['category_id'] == 6:
                    ann['category_id'] -= 1
                gt_label.append(ann['category_id']-1)

            multilabel_jsondata['data_list'].append({
                "img_path": imgitem["file_name"],
                "gt_label": list(set(gt_label))
            })
            pn_cnt[imgitem['diagnose']] += 1
        print(f'{mode} pn_cnt: {pn_cnt}')

        with open(f'{json_save_dir}/multilabel_{mode}.json', 'w', encoding='utf-8') as f:
            json.dump(multilabel_jsondata, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
    statistic()

'''
WINDOW_SIZE = 400
train pn_cnt: [28702, 32532]
val pn_cnt: [3572, 3678]

+--------------------------------------+
|                train                 |
+------+--------+------+-------+-------+
| AGC  | ASC-US | LSIL | ASC-H |  HSIL |
+------+--------+------+-------+-------+
| 5739 |  3156  | 2929 |  4341 | 19498 |
+------+--------+------+-------+-------+
+------------------------------------+
|                test                |
+-----+--------+------+-------+------+
| AGC | ASC-US | LSIL | ASC-H | HSIL |
+-----+--------+------+-------+------+
| 770 |  357   | 342  |  464  | 2063 |
+-----+--------+------+-------+------+
'''