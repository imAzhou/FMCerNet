from glob import glob
import json
from cerwsi.utils import read_json_anno
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import itertools
import numpy as np
import pandas as pd
from prettytable import PrettyTable

plt.rcParams['axes.unicode_minus'] = False

def flatten_list(nested_list):
    """递归展开任意层嵌套的列表"""
    for item in nested_list:
        if isinstance(item, list):
            yield from flatten_list(item)
        else:
            yield str(item)

def main():
    JFSW_1_jsons = glob('/medical-data_NB/data/cervix/JFSW/阳性json/*.json')
    JFSW_2_jsons = glob('/medical-data_NB/data/cervix/JFSW_1109/**/json/*.json')
    all_jsons_path = [*JFSW_1_jsons, *JFSW_2_jsons]

    total_desc_items = defaultdict(list)
    for jsonpath in tqdm(all_jsons_path, ncols=80):
        annos = read_json_anno(jsonpath)
        
        for ann_ in annos:
            ann_clsname = ann_.get('sub_class')
            hierarchical_annotation = ann_.get('hierarchical_annotation', [])
            if ann_clsname is not None and hierarchical_annotation:
                total_desc_items[ann_clsname].append(hierarchical_annotation)
    
    save_dir = 'data_resource/lesion_desc'
    os.makedirs(save_dir, exist_ok=True, mode=0o777)
    # 保存到 txt 文件（去重）
    summary_desc = []
    for clsname, desc_lists in total_desc_items.items():
        unique_texts = set()
        for desc in desc_lists:
            flat_list = list(set(flatten_list(desc)))
            flat_text = ','.join(flat_list).strip()
            if flat_text:
                unique_texts.add(flat_text)

        summary_desc.append(f'{clsname}')
        
        save_path = os.path.join(save_dir, f"{clsname}.txt")
        with open(save_path, 'w', encoding='utf-8') as f:
            for text in sorted(unique_texts):
                f.write(text + '\n')

        print(f"已保存 {clsname}.txt（共 {len(unique_texts)} 条唯一描述）")

    print(f"\n✅ 已保存 {len(total_desc_items)} 个类别的描述文件到：{save_dir}")

def attri_count():
    JFSW_1_jsons = glob('/medical-data/data/cervix/JFSW/阳性json/*.json')
    JFSW_2_jsons = glob('/medical-data/data/cervix/JFSW_1109/**/json/*.json')
    all_jsons_path = [*JFSW_1_jsons, *JFSW_2_jsons]
    tgt_clsname = ['GEC', 'NILM','AGC','AGC-FN','AGC-N','AGC-NOS','ASC-US','LSIL','ASC-H','HSIL']
    del_attri = ['单个细胞','成团细胞','GEC', 'HSIL', '阴性', 'AGC-NOS', 'Inflammatory', 'ASC-US', 'LSIL', 'AGC', 'NILM', 'ASC-H', '阳性', 'AGC-FN', 'AGC-N', '核仁增大/多核仁']

    desc_cell_list = []
    for jsonpath in tqdm(all_jsons_path, ncols=80):
        annos = read_json_anno(jsonpath)
        for ann_ in annos:
            ann_clsname = ann_.get('sub_class')
            if ann_clsname not in tgt_clsname:
                continue
            region = ann_.get('region')
            w,h = region['width'],region['height']
            if w <=20 or h<=20:
                continue
            hierarchical_annotation = ann_.get('hierarchical_annotation', [])
            desc_list = []
            for desc in list(set(flatten_list(hierarchical_annotation))):
                if desc not in del_attri:
                    desc_list.append(desc)

            if ann_clsname not in ['GEC', 'NILM'] and not desc_list:
                continue
            desc_cell_list.append({
                'clsname': ann_clsname,
                'desc_list': desc_list
            }) 

    print(f'细胞实例总数: {len(desc_cell_list)}')
    with open(instance_savepath, 'w', encoding='utf-8') as f:
        json.dump(desc_cell_list, f, ensure_ascii=False)

def generate_analysis(data, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # -----------------------------
    # 类别计数
    # -----------------------------
    tgt_clsname = ['NILM','ASC-US','LSIL','ASC-H','HSIL','GEC','AGC','AGC-FN','AGC-N','AGC-NOS']
    tgt_desc = ["核增大到2.5-3倍", "核增大大于3倍","核浆比大","核增大","异形双核或多核","核轻度深染","核染色质轻度深染","核染色质加深","核染色质呈细颗粒状","核染色质呈粗颗粒状","核膜不规则","胞浆有空泡","不完全挖空化细胞","挖空细胞","非典型角化细胞","葡萄干核型","核异形","核异形性增强","核大小不一","核位上移","失去极向，排列紊乱","羽毛状排列","细胞团呈三维簇团结构","栅栏状排列紊乱","乳头状排列紊乱","菊形团排列","腺腔样排列","细胞的大小形态不一，排列紊乱","核深染拥挤"]
    class_counter = Counter([d["ori_clsname"] for d in data])
    class_sorted = dict(sorted(class_counter.items(), key=lambda x: x[1], reverse=True))

    # 保存 txt（PrettyTable 美化）
    tb_class = PrettyTable()
    tb_class.field_names = ["Class Name", "Instance Count"]
    for cls in tgt_clsname:
        cnt = class_sorted[cls]
        tb_class.add_row([cls, cnt])
    with open(os.path.join(save_dir, "class_instance_count.txt"), "w", encoding="utf-8") as f:
        f.write(str(tb_class))

    # -----------------------------
    # 属性计数
    # -----------------------------
    all_attrs = list(itertools.chain.from_iterable([d["jfsw_desc"] for d in data]))
    attr_counter = Counter(all_attrs)
    attr_sorted = dict(sorted(attr_counter.items(), key=lambda x: x[1], reverse=True))

    # 保存 txt（PrettyTable 美化）
    tb_attr = PrettyTable()
    tb_attr.field_names = ["Attribute Name", "Instance Count"]
    for attr, cnt in attr_sorted.items():
        tb_attr.add_row([attr, cnt])
    with open(os.path.join(save_dir, "attribute_instance_count.txt"), "w", encoding="utf-8") as f:
        f.write(str(tb_attr))

    cls_attri_cnt = defaultdict(list)
    for d in data:
        cls_attri_cnt[d['sub_class']].append(d['attr_v'])
    tb_cacnt = PrettyTable()
    first_class = next(iter(cls_attri_cnt))
    num_attrs = len(cls_attri_cnt[first_class][0])
    field_names = ["Class"] + [f"attr_{i}" for i in range(num_attrs)]
    tb_cacnt.field_names = field_names
    for sub_class in tgt_clsname:
        if 'AGC' in sub_class:
            sub_class = 'AGC'
        attr_list = cls_attri_cnt[sub_class]
        indices = list(zip(*attr_list))
        row_values = []
        for values in indices:
            unique_vals = sorted(list(set(values)))
            row_values.append(str(tuple(unique_vals)))
        tb_cacnt.add_row([sub_class] + row_values)
    tb_cacnt.align["Class"] = "l"
    with open(os.path.join(save_dir, "cls_attri_dist.txt"), "w", encoding="utf-8") as f:
        f.write(tb_cacnt.get_string())

    # -----------------------------
    # 属性共现矩阵（Excel）
    # -----------------------------
    co_mat = pd.DataFrame(0, index=tgt_desc, columns=tgt_desc)
    for d in data:
        desc = list(set(d["jfsw_desc"]))
        for i in range(len(desc)):
            for j in range(i, len(desc)):
                co_mat.loc[desc[i], desc[j]] += 1
                if i != j:
                    co_mat.loc[desc[j], desc[i]] += 1

    co_mat.to_excel(os.path.join(save_dir, "attribute_co_occurrence.xlsx"))
    # 2. 转为共现比例
    co_ratio = co_mat.copy()
    for a in tgt_desc:
        for b in tgt_desc:
            if a != b:
                co_ratio.loc[a, b] = co_mat.loc[a, b] / min(attr_counter[a], attr_counter[b])
            else:
                co_ratio.loc[a, b] = 1.0  # 自己与自己

    # 3. 保存 Excel
    co_ratio.to_excel(os.path.join(save_dir, "attribute_correlation.xlsx"))

    # -----------------------------
    # 类别-属性矩阵（Excel）
    # -----------------------------
    class_attr_mat = pd.DataFrame(0, index=tgt_clsname, columns=tgt_desc)
    for d in data:
        cls = d["ori_clsname"]
        for a in d["jfsw_desc"]:
            class_attr_mat.loc[cls, a] += 1

    class_attr_mat.to_excel(os.path.join(save_dir, "class_attribute_correlation.xlsx"))

    print(f"所有统计图、txt和Excel已保存到 {save_dir}")


if __name__ == "__main__":
    # instance_savepath = 'data_resource/cell_attri/statistic_result/cell_inst_desc.json'
    # main()
    # attri_count()

    instance_savepath = 'data_resource/cell_attri/cell_inst.json'
    with open(instance_savepath, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    cell_list = []
    for pidlist in json_data.values():
        cell_list.extend(pidlist)
    generate_analysis(cell_list, 'statistic_results/attribute_analyze')
