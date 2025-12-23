import json

from tqdm import tqdm


json_path = 'data_resource/cell_attri/cell_inst_named.json'

with open(json_path, 'r', encoding='utf-8') as f:
    json_data = json.load(f)

tgt_desc = ['核大小不一','失去极向，排列紊乱','羽毛状排列','细胞团呈三维簇团结构', '栅栏状排列紊乱', '乳头状排列紊乱', '菊形团排列', '腺腔样排列']
total_error_cnt = 0
tgt_num = 0
for tile_list in tqdm(json_data.values(), ncols=80):
    for tileitem in tile_list:
        cnt = 0
        if '菊形团排列' not in tileitem['jfsw_desc']:
            continue
        tgt_num += 1
        for desc in tileitem['jfsw_desc']:
            if desc in tgt_desc:
                cnt += 1
        if cnt > 1:
            total_error_cnt += 1
        else:
            print(tileitem['jfsw_desc'])

print(tgt_num)
print(total_error_cnt)