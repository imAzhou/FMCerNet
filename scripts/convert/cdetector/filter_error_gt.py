import json


def main():
    root_dir = 'data_resource/ComparisonDetectorDataset'
    for jsosmode in ['train','test']:
        with open(f'{root_dir}/{jsosmode}.json', 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        new_annotations = []
        for ann in json_data['annotations']:
            _,_,w,h = ann['bbox']
            if w < 10 or h<10:
                continue
            new_annotations.append(ann)
        json_data['annotations'] = new_annotations

        with open(f'{root_dir}/{jsosmode}_filter_error.json', 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False)

if __name__ == "__main__":
    main()