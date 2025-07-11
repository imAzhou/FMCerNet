import json
from pycocotools.coco import COCO
from tqdm import tqdm
import random

def gene_mini_json():
    annojson_save_dir = f'{root_dir}/annofiles'

    for mode in ['train', 'val']:
    # for mode in ['train']:
        annFile = f'{annojson_save_dir}/OD_instances_{mode}.json'
        with open(annFile, 'r') as f:
            OD_anno_json = json.load(f)
        coco = COCO(annFile)

        multilabel_jsonpath = f'{annojson_save_dir}/{mode}_patches.json'
        with open(multilabel_jsonpath, 'r') as f:
            multilabel_anno_json = json.load(f)
        filename2idx = {}
        for idx,item in enumerate(multilabel_anno_json):
            filename2idx[item['filename']] = idx

        mini_positive,mini_negative = [],[]
        # 遍历所有图像
        for imgItem in tqdm(OD_anno_json['images'], ncols=80):
            imgId = imgItem['id']
            imgFilename = imgItem['file_name'].split('/')[1]
            annIds = coco.getAnnIds(imgIds=imgId)
            anns = coco.loadAnns(annIds)
            if len(anns) == 0 and random.random() < 0.25:
                mini_negative.append(multilabel_anno_json[filename2idx[imgFilename]])
            for ann in anns:
                bbox = ann['bbox']
                area = bbox[2] * bbox[3]  # 宽 * 高
                if area < 64 * 64:
                    mini_positive.append(multilabel_anno_json[filename2idx[imgFilename]])
                    break
        print(f'{mode}: positive: {len(mini_positive)}, negative: {len(mini_negative)}')
        mini_multilabel_json = [*mini_negative, *mini_positive]
        random.shuffle(mini_multilabel_json)
        with open(f'{annojson_save_dir}/mini_{mode}_patches.json', 'w') as f:
            json.dump(mini_multilabel_json, f)
        

if __name__ == '__main__':
    root_dir = '/x22201018/datasets/CervicalDatasets/ComparisonDetectorDataset'
    # root_dir = '/c22073/zly/datasets/CervicalDatasets/ComparisonDetectorDataset'
    # root_dir = '/disk/medical_datasets/cervix/ComparisonDetectorDataset'
    
    gene_mini_json()
