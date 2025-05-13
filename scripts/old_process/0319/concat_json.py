import json
from tqdm import tqdm
from prettytable import PrettyTable

classes = ['NILM', 'ASC-US', 'LSIL', 'ASC-H', 'HSIL', 'AGC']
# 类别映射关系
RECORD_CLASS = {
    'ASC-US': 'ASC-US',
    'LSIL': 'LSIL',
    'ASC-H': 'ASC-H',
    'HSIL': 'HSIL',
    'SCC': 'HSIL',
    'AGC-N': 'AGC',
    'AGC': 'AGC',
    'AGC-NOS': 'AGC',
    'AGC-FN': 'AGC',
}

if __name__ == '__main__':
    
    json_root = 'data_resource/0319/annofiles'
    for mode in ['train','val']:
        datalist = []
        classes_cnt = [0]*len(classes)
        for clstype in ['posslide','negslide']:
            with open(f'{json_root}/{mode}_{clstype}_patches.json', 'r') as f:
                pdata = json.load(f)
            
            for item in tqdm(pdata,ncols=80):
                for patchInfo in item['patch_list']:   # for in each slide patch list
                    prefix = ''
                    if clstype == 'negslide':
                        prefix = 'Neg'
                    else:
                        prefix = 'Pos'
                    patchInfo['prefix'] = prefix
                    patchInfo['clsnames'] = [RECORD_CLASS[i] for i in patchInfo['clsnames']]
                    patchInfo['clsid'] = [classes.index(i) for i in patchInfo['clsnames']]
                    datalist.append(patchInfo)

                    if len(patchInfo['bboxes']) == 0:
                        classes_cnt[0] += 1
                    else:
                        for idx in patchInfo['clsid']:
                            classes_cnt[idx] += 1
        
        result_table = PrettyTable()
        result_table.field_names = classes + ['Pos', 'Total']
        result_table.add_row(classes_cnt + [len(datalist)-classes_cnt[0], len(datalist)])
        print(result_table)
    
        with open(f'{json_root}/{mode}_patches_v0319.json', 'w') as f:
            json.dump(datalist, f)
            
'''
train:
+-------+--------+-------+-------+-------+-------+-------+-------+
|  NILM | ASC-US |  LSIL | ASC-H |  HSIL |  AGC  |  Pos  | Total |
+-------+--------+-------+-------+-------+-------+-------+-------+
| 45305 | 37929  | 12101 | 38735 | 16538 | 21494 | 54028 | 99333 |
+-------+--------+-------+-------+-------+-------+-------+-------+

val:
+-------+--------+------+-------+------+------+-------+-------+
|  NILM | ASC-US | LSIL | ASC-H | HSIL | AGC  |  Pos  | Total |
+-------+--------+------+-------+------+------+-------+-------+
| 11551 | 10072  | 2364 | 13806 | 4444 | 9874 | 14199 | 25750 |
+-------+--------+------+-------+------+------+-------+-------+
'''