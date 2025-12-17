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
    del_attri = ['单个细胞','成团细胞','GEC', 'HSIL', '阴性', 'AGC-NOS', 'Inflammatory', 'ASC-US', 'LSIL', 'AGC', 'NILM', 'ASC-H', '阳性', 'AGC-FN', 'AGC-N']

    desc_cell_list = []
    for jsonpath in tqdm(all_jsons_path, ncols=80):
        annos = read_json_anno(jsonpath)
        for ann_ in annos:
            ann_clsname = ann_.get('sub_class')
            if ann_clsname not in tgt_clsname:
                continue
            
            hierarchical_annotation = ann_.get('hierarchical_annotation', [])
            desc_list = []
            for desc in list(set(flatten_list(hierarchical_annotation))):
                if desc not in del_attri:
                    desc_list.append(desc)
            
            if '成团细胞' in desc_list and '单个细胞' in desc_list:     # 16条脏数据，同时被标注成 单个和成团 
                continue

            if ann_clsname is not None and desc_list:
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
    class_counter = Counter([d["clsname"] for d in data])
    class_sorted = dict(sorted(class_counter.items(), key=lambda x: x[1], reverse=True))

    # 绘制柱状图
    plt.figure(figsize=(10, 6))
    x = list(class_sorted.keys())
    y = list(class_sorted.values())
    bars = plt.bar(x, y)
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 str(bar.get_height()), ha='center', va='bottom')
    plt.xlabel("Class Name")
    plt.ylabel("Instance Count")
    plt.title("Class Instance Count")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "class_instance_count.png"))
    plt.close()

    # 保存 txt（PrettyTable 美化）
    tb_class = PrettyTable()
    tb_class.field_names = ["Class Name", "Instance Count"]
    for cls, cnt in class_sorted.items():
        tb_class.add_row([cls, cnt])
    with open(os.path.join(save_dir, "class_instance_count.txt"), "w", encoding="utf-8") as f:
        f.write(str(tb_class))

    # -----------------------------
    # 属性计数
    # -----------------------------
    all_attrs = list(itertools.chain.from_iterable([d["desc_list"] for d in data]))
    attr_counter = Counter(all_attrs)
    attr_sorted = dict(sorted(attr_counter.items(), key=lambda x: x[1], reverse=True))

    # 绘制柱状图
    plt.figure(figsize=(12, 6))
    attrs = list(attr_sorted.keys())
    counts = list(attr_sorted.values())
    bars = plt.bar(attrs, counts)
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 str(bar.get_height()), ha='center', va='bottom')
    plt.xlabel("Attribute")
    plt.ylabel("Instance Count")
    plt.title("Attribute Instance Count")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "attribute_instance_count.png"))
    plt.close()

    # 保存 txt（PrettyTable 美化）
    tb_attr = PrettyTable()
    tb_attr.field_names = ["Attribute Name", "Instance Count"]
    for attr, cnt in attr_sorted.items():
        tb_attr.add_row([attr, cnt])
    with open(os.path.join(save_dir, "attribute_instance_count.txt"), "w", encoding="utf-8") as f:
        f.write(str(tb_attr))

    # -----------------------------
    # 属性共现矩阵（Excel）
    # -----------------------------
    attr_list = list(attr_counter.keys())
    co_mat = pd.DataFrame(0, index=attr_list, columns=attr_list)

    for d in data:
        desc = list(set(d["desc_list"]))
        for i in range(len(desc)):
            for j in range(i, len(desc)):
                co_mat.loc[desc[i], desc[j]] += 1
                if i != j:
                    co_mat.loc[desc[j], desc[i]] += 1

    co_mat.to_excel(os.path.join(save_dir, "attribute_co_occurrence.xlsx"))
    # 2. 转为共现比例
    co_ratio = co_mat.copy()
    for a in attr_list:
        for b in attr_list:
            if a != b:
                co_ratio.loc[a, b] = co_mat.loc[a, b] / min(attr_counter[a], attr_counter[b])
            else:
                co_ratio.loc[a, b] = 1.0  # 自己与自己

    # 3. 保存 Excel
    co_ratio.to_excel(os.path.join(save_dir, "attribute_correlation.xlsx"))

    # -----------------------------
    # 类别-属性矩阵（Excel）
    # -----------------------------
    class_attr_mat = pd.DataFrame(0, index=class_counter.keys(), columns=attr_list)
    for d in data:
        cls = d["clsname"]
        for a in d["desc_list"]:
            class_attr_mat.loc[cls, a] += 1

    class_attr_mat.to_excel(os.path.join(save_dir, "class_attribute_correlation.xlsx"))

    print(f"所有统计图、txt和Excel已保存到 {save_dir}")

if __name__ == "__main__":
    instance_savepath = 'data_resource/lesion_desc/cell_inst_desc.json'
    main()
    # attri_count()

    # with open(instance_savepath, 'r', encoding='utf-8') as f:
    #     json_data = json.load(f)
    # generate_analysis(json_data, 'statistic_results/attribute_analyze')
