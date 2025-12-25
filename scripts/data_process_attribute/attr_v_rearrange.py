import json
import os
import random
from collections import Counter, defaultdict
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

def visual_sample(celllist):
    filename_prefix = 'mn'
    save_dir = 'data_resource/cell_attri/sample_imgs/LSIL'
    os.makedirs(save_dir, exist_ok=True, mode=0o777)
    with open('data_resource/cell_attri/config_attri.json', 'r', encoding='utf-8') as f:
        attr_config = json.load(f)

    try:
        font_path = "data_resource/SimHei.ttf"
        font_size = 20
        font = ImageFont.truetype(font_path, font_size)
        title_font = ImageFont.truetype(font_path, font_size + 4) # 标题稍大
    except IOError:
        print("警告：未找到指定中文字体，尝试使用默认字体（可能无法显示中文）。")
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()

    for sample_item in celllist:
        imgpath = f'data_resource/cell_attri/cell_inst/images/{sample_item["filename"]}'
        jfsw_desc = [sample_item["sub_class"], *sample_item["jfsw_desc"]]
        attr_desc = [f"{attr_config[idx]['attr_name']}: {attr_config[idx]['children'][v]}" for idx,v in enumerate(sample_item["attr_v"])]

        # 合并文本用于计算高度，添加一些标题区分
        text_lines = ["【JFSW 描述】"] + jfsw_desc + ["", "【属性 描述】"] + attr_desc
        
        try:
            # 2. 打开原始图像
            with Image.open(imgpath) as img:
                w, h = img.size
                
                # 3. 设置右侧文字区域的参数
                text_padding = 20    # 文字离图片的间距
                text_area_width = 400 # 预留给文字的宽度 (根据文字长度可调整)
                line_spacing = 5     # 行间距
                
                # 计算需要的总高度 (防止文字比图片长)
                # 简单的估算：行数 * (字体大小 + 行间距) + 上下边距
                text_total_height = len(text_lines) * (font_size + line_spacing) + 40
                
                # 新图像的宽 = 原宽 + 间距 + 文字区宽
                new_w = w + text_padding + text_area_width
                # 新图像的高 = 原高 和 文字总高 中的最大值
                new_h = max(h, text_total_height)
                
                # 4. 创建新画布 (白色背景)
                new_img = Image.new('RGB', (new_w, new_h), (255, 255, 255))
                
                # 5. 粘贴原图 (垂直居中或者顶部对齐，这里选择顶部对齐)
                new_img.paste(img, (0, 0))
                
                # 6. 绘制文字
                draw = ImageDraw.Draw(new_img)
                current_y = 20 # 初始 Y 坐标
                text_start_x = w + text_padding
                
                for line in text_lines:
                    # 判断是标题还是普通内容，使用不同颜色或字体
                    if line.startswith("【"):
                        curr_font = title_font
                        fill_color = (0, 0, 139) # 深蓝色
                    else:
                        curr_font = font
                        fill_color = (0, 0, 0)   # 黑色
                    
                    draw.text((text_start_x, current_y), line, font=curr_font, fill=fill_color)
                    # 更新 Y 坐标
                    bbox = draw.textbbox((text_start_x, current_y), line, font=curr_font)
                    text_height = bbox[3] - bbox[1]
                    current_y += text_height + line_spacing

                # 7. 保存图像
                save_path = os.path.join(save_dir, filename_prefix+sample_item["filename"])
                new_img.save(save_path)
                print(f"Saved: {save_path}")
                
        except FileNotFoundError:
            print(f"Error: 找不到图片 {imgpath}")
        except Exception as e:
            print(f"Error processing {imgpath}: {e}")

def map_desc2vec(desc_list, sub_class):
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
    gland_flag = 0 if sub_class in ['GEC', 'AGC'] else 1
    attri_vec[-1] = gland_flag
    return attri_vec

def statistic():
    with open('data_resource/cell_attri/config_attri.json', 'r', encoding='utf-8') as f:
        attr_config = json.load(f)
    attr_number = [len(i['children']) for i in attr_config]
    out_dir = 'data_resource/cell_attri/statistic_result_rerearrange'
    os.makedirs(out_dir, exist_ok=True)

    class_txt = os.path.join(out_dir, 'class_dist.txt')
    area_txt  = os.path.join(out_dir, 'area_dist.txt')
    attr_txt  = os.path.join(out_dir, 'attr_dist.txt')

    with open(class_txt, 'w', encoding='utf-8') as f_cls, \
         open(area_txt,  'w', encoding='utf-8') as f_area, \
         open(attr_txt,  'w', encoding='utf-8') as f_attr:

        for mode in ['train', 'val']:
            with open(f'data_resource/cell_attri/cell_inst/rere_{mode}_cellinst.json',
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



def main():
    for mode in ['train', 'val']:
        with open(f'data_resource/cell_attri/cell_inst/{mode}_cellinst.json', 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        new_json_data = []
        for cellitem in tqdm(json_data, ncols=80):
            cellitem['attr_v'] = map_desc2vec(cellitem['jfsw_desc'],cellitem['sub_class'])
            new_json_data.append(cellitem)
        with open(f'data_resource/cell_attri/cell_inst/rere_{mode}_cellinst.json', 'w', encoding='utf-8') as f:
            json.dump(new_json_data, f, ensure_ascii=False)
    # random.shuffle(tgt_celllist)
    # visual_sample(tgt_celllist[:10])

if __name__ == "__main__":
    main()
    statistic()