import json
from tqdm import tqdm
from collections import Counter, defaultdict
import os

def filter_tgt_attrv(data):
    attrv2clsname = defaultdict(list)
    for item in tqdm(data, ncols=80):
        new_attrv = item['attr_v'][:-1]
        sub_class = item['sub_class']
        gland_flag = 0 if sub_class in ['AGC', 'GEC'] else 1
        new_attrv.append(gland_flag)
        attr_v_key = str(new_attrv)
        attrv2clsname[attr_v_key].append(item)
    conflict_cnt = 0
    include_cell_cnt = 0
    lines, new_data = [],[]
    for k,v in attrv2clsname.items():
        unique_v = list(set([i['sub_class'] for i in v]))
        if len(unique_v)>1:
            strprint = f'\nattr_v-{k}\n'
            clsname2desc = defaultdict(list)
            for i in v:
                clsname2desc[i['sub_class']].append(','.join(sorted(i['jfsw_desc'])))
            for clsname,desc_list in clsname2desc.items():
                unique_desc = list(set(desc_list))
                strprint += f'\t{clsname}, jfsw_desc:\n'
                for descitem in unique_desc:
                    strprint += f'\t\t{descitem}\n'
            
            lines.append(strprint)
            conflict_cnt += 1
            include_cell_cnt += len(v)
        else:
            new_data.extend(v)
    with open('data_resource/cell_attri/statistic_result/attrv_conflict.txt', 'w') as f:
        f.writelines(lines) 
    
    print(len(attrv2clsname.keys()))
    print(conflict_cnt)
    print(include_cell_cnt)

    return new_data

def statistic(attr_config, out_dir):
    attr_number = [len(i['children']) for i in attr_config]
    class_txt = os.path.join(out_dir, 'class_dist.txt')
    area_txt  = os.path.join(out_dir, 'area_dist.txt')
    attr_txt  = os.path.join(out_dir, 'attr_dist.txt')

    with open(class_txt, 'w', encoding='utf-8') as f_cls, \
         open(area_txt,  'w', encoding='utf-8') as f_area, \
         open(attr_txt,  'w', encoding='utf-8') as f_attr:

        for mode in ['train', 'train_hs0', 'val']:
            with open(f'data_resource/cell_attri/cell_inst/filter_{mode}_cellinst.json',
                      'r', encoding='utf-8') as f:
                json_data = json.load(f)

            # ======================
            # 1. 类别统计
            # ======================
            clsname_instcnt = Counter()

            # ======================
            # 2. area 统计
            # ======================
            area_cnt = {
                'small (<64x64)': 0,
                'medium (64x64~128x128)': 0,
                'large (>=128x128)': 0
            }

            # ======================
            # 3. 属性统计（每个 mode 独立）
            # ======================
            attr_cnt = [[0 for _ in range(v)] for v in attr_number]

            for cellItem in tqdm(json_data, desc=f'{mode}', ncols=80):
                # 类别
                clsname_instcnt[cellItem['sub_class']] += 1

                # area
                area = cellItem['area']
                if area < 64 * 64:
                    area_cnt['small (<64x64)'] += 1
                elif area < 128 * 128:
                    area_cnt['medium (64x64~128x128)'] += 1
                else:
                    area_cnt['large (>=128x128)'] += 1

                # 属性
                for idx, v in enumerate(cellItem['attr_v']):
                    attr_cnt[idx][v] += 1

            total_inst = sum(clsname_instcnt.values())

            # =====================================================
            # 写 1：类别统计
            # =====================================================
            f_cls.write(f"{mode.upper()} 类别统计\n")
            f_cls.write("=" * 60 + "\n")
            f_cls.write(f"总实例数: {total_inst}\n")
            f_cls.write(f"类别数: {len(clsname_instcnt)}\n\n")

            for cname, cnt in sorted(clsname_instcnt.items(),
                                     key=lambda x: x[1],
                                     reverse=True):
                pct = cnt / total_inst * 100
                f_cls.write(f"{cname:<20}: {cnt:>6} ({pct:>6.2f}%)\n")

            f_cls.write("\n\n")

            # =====================================================
            # 写 2：area 分布
            # =====================================================
            f_area.write(f"{mode.upper()} Area 分布\n")
            f_area.write("=" * 60 + "\n")
            f_area.write(f"总实例数: {total_inst}\n\n")

            for k, v in area_cnt.items():
                pct = v / total_inst * 100 if total_inst > 0 else 0
                f_area.write(f"{k:<30}: {v:>6} ({pct:>6.2f}%)\n")

            f_area.write("\n\n")

            # =====================================================
            # 写 3：属性值分布
            # =====================================================
            f_attr.write(f"{mode.upper()} 属性值分布\n")
            f_attr.write("=" * 60 + "\n")

            for attr_idx, (dist, max_val) in enumerate(zip(attr_cnt, attr_number)):
                attr_item = attr_config[attr_idx]
                f_attr.write(f"属性 {attr_item['attr_name']} (0-{max_val - 1}) (default: {attr_item['default_value']})\n")
                total = sum(dist)

                for val_idx, cnt in enumerate(dist):
                    pct = cnt / total * 100 if total > 0 else 0
                    f_attr.write(f"  值 {val_idx}({attr_item['children'][val_idx]}): {cnt:>6} ({pct:>6.2f}%)\n")
                f_attr.write("-" * 50 + "\n")

            f_attr.write("\n\n")

    print("统计完成，结果已保存至:", out_dir)


def remap_attrv(item):
    new_attrv = item['attr_v'][:-1]
    sub_class = item['sub_class']
    gland_flag = 0 if sub_class in ['AGC', 'GEC'] else 1
    new_attrv.append(gland_flag)
    item['attr_v'] = new_attrv
    return item

if __name__ == "__main__":
    # instance_savepath = 'data_resource/cell_attri/cell_inst_named.json'
    # with open(instance_savepath, 'r', encoding='utf-8') as f:
    #     json_data = json.load(f)
    # cell_list = []
    # for pidlist in json_data.values():
    #     cell_list.extend(pidlist)
    
    # new_cell_inst = filter_tgt_attrv(cell_list)
    # valid_filenames = [i["filename"] for i in new_cell_inst]
    # for mode in ['train', 'train_hs0', 'val']:
    #     jsonpath = f'data_resource/cell_attri/cell_inst/{mode}_cellinst.json'
    #     with open(jsonpath, 'r', encoding='utf-8') as f:
    #         origin_datalist = json.load(f)
    #     new_datalist = [remap_attrv(item) for item in origin_datalist if item['filename'] in valid_filenames]
    #     new_jsonpath = f'data_resource/cell_attri/cell_inst/filter_{mode}_cellinst.json'
    #     with open(new_jsonpath, 'w', encoding='utf-8') as f:
    #         json.dump(new_datalist, f, ensure_ascii=False)
    
    # out_dir = 'data_resource/cell_attri/statistic_result_filter'
    # os.makedirs(out_dir, exist_ok=True)
    # with open('data_resource/cell_attri/configs/attri_defined.json', 'r', encoding='utf-8') as f:
    #     attr_config = json.load(f)
    # attr_config[-1] = {
    #     "attr_name": "细胞类型",
    #     "tag": "cellType",
    #     "default_value": 0,
    #     "children": ["Gland", "Non-Gland"]
    # }
    # statistic(attr_config, out_dir)
    
    jsonpath = f'data_resource/cell_attri/cell_inst/filter_train_cellinst.json'
    with open(jsonpath, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    cls_attrlist = defaultdict(list)
    for cellitem in tqdm(train_data, ncols=80):
        cls_attrlist[cellitem["sub_class"]].append(','.join([str(i) for i in cellitem['attr_v']]))
    
    cls_attrset = {}
    for clsname, attrlist in cls_attrlist.items():
        unique_attrlist = list(set(attrlist))
        cls_attrset[clsname] = []
        for attrlist in unique_attrlist:
            cls_attrset[clsname].append([int(i) for i in attrlist.split(',')])
    with open('data_resource/cell_attri/configs/cls_attrset.json', 'w', encoding='utf-8') as f:
        json.dump(cls_attrset, f, ensure_ascii=False)
    
    
