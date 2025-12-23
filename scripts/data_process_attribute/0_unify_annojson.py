import json
from cerwsi.utils import read_json_anno
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
from cerwsi.utils.tools import remap_points


RECORD_CLASS = {
    'NILM': 'NILM',
    'GEC': 'GEC',
    'AGC': 'AGC',
    'AGC-N': 'AGC',
    'AGC-NOS': 'AGC',
    'AGC-FN': 'AGC',
    'ASC-US': 'ASC-US',
    'LSIL': 'LSIL',
    'ASC-H': 'ASC-H',
    'HSIL': 'HSIL',
}


def flatten_list(nested_list):
    """递归展开任意层嵌套的列表"""
    for item in nested_list:
        if isinstance(item, list):
            yield from flatten_list(item)
        else:
            yield str(item)

def check_inst(del_attri, ann):
    ann = remap_points(ann)
    if ann is None:
        return None
    sub_class = ann.get('sub_class')
    if sub_class not in RECORD_CLASS.keys():
        return  None
    region = ann.get('region')
    w,h = region['width'],region['height']
    if w <=20 or h<=20:
        return  None
    
    desc_list = []
    if sub_class in ['NILM', 'GEC']:
        ann['desc_list'] = desc_list
        return ann
    
    hierarchical_annotation = ann.get('hierarchical_annotation', [])
    for desc in list(set(flatten_list(hierarchical_annotation))):
        if desc not in del_attri:
            desc_list.append(desc)
    if not desc_list:
        return None
    ann['desc_list'] = desc_list 
    return ann

def map_desc2vec(desc_list):
    # init value
    with open('data_resource/cell_attri/config_attri.json', 'r', encoding='utf-8') as f:
        attri_cfg = json.load(f)
    attri_vec = [i['default_value'] for i in attri_cfg]
    with open('data_resource/cell_attri/config_desc.json', 'r', encoding='utf-8') as f:
        desc_cfg = json.load(f)
    invalid_items = [item for item in desc_list if item not in desc_cfg]
    assert len(invalid_items) == 0, f"desc_list 中有 {len(invalid_items)} 个不存在的 key: {invalid_items}"
    
    sorted_desc_list = sorted(  # priority 从大到小排序，越模糊的描述 priority 值越大
        desc_list, reverse=True,
        key=lambda item: desc_cfg[item]["priority"]
    )
    for desc in sorted_desc_list:
        update_idx = desc_cfg[desc]['update_idx']
        update_value = desc_cfg[desc]['update_value']
        for idx,value in zip(update_idx, update_value):
            attri_vec[idx] = value
    
    return attri_vec

def main():
    del_attri = ['阴性', '阳性', 'GEC', 'NILM', 'Inflammatory', '核仁增大/多核仁', '单个细胞', '成团细胞',
                 'HSIL', 'AGC-NOS', 'ASC-US', 'LSIL', 'AGC', 'ASC-H', 'AGC-FN', 'AGC-N', 'SCC']

    abandon_data = pd.read_csv('data_resource/group_csv/abandon.csv')
    JFSW_1_data = pd.read_csv('data_resource/group_csv/JFSW_1.csv')
    JFSW_2_data = pd.read_csv('data_resource/group_csv/JFSW_2.csv')
    JFSW_data = pd.concat([JFSW_1_data, JFSW_2_data])
    JFSW_data = JFSW_data[~JFSW_data['patientId'].isin(abandon_data['patientId'])]

    total_inst_items = defaultdict(list)
    for row in tqdm(JFSW_data.itertuples(index=False), total=len(JFSW_data), ncols=80):
        if type(row.anno_path) != str or not row.anno_path:
            continue
        patientId = row.patientId
        annos = read_json_anno(row.anno_path)
        for ann_ in annos:
            ann = check_inst(del_attri, ann_)
            if ann is None:
                continue
            sub_class = ann.get('sub_class')
            region = ann.get('region')
            x,y = region['x'],region['y']
            w,h = region['width'],region['height']
            
            desc_list = ann['desc_list']
            attr_v = map_desc2vec(desc_list)
            total_inst_items[patientId].append({
                'patientId': patientId,
                'sub_class': RECORD_CLASS[sub_class],
                'bbox': [x,y,x+w,y+h],
                'area': w*h,
                'jfsw_desc': desc_list,
                'attr_v': attr_v
            })
    
    with open(instance_savepath, 'w', encoding='utf-8') as f:
        json.dump(total_inst_items, f, ensure_ascii=False)



if __name__ == "__main__":
    instance_savepath = 'data_resource/cell_attri/cell_inst.json'
    main()

    
