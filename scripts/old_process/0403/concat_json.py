import json
from tqdm import tqdm
from prettytable import PrettyTable

classes = ['NILM', 'AGC', 'ASC-US', 'LSIL', 'ASC-H', 'HSIL']
# 类别映射关系
RECORD_CLASS = {
    'NILM': 'NILM',
    'GEC': 'NILM',
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
    
    json_root = 'data_resource/0403/annofiles'
    for mode in ['train','val']:
        datalist = []
        classes_cnt = [0,0]
        for clstype in ['posslide','negslide']:
            json_path = f'{json_root}/{mode}_{clstype}_patches'
            if clstype == 'posslide':
                json_path += '_filtered'
            with open(f'{json_path}.json', 'r') as f:
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
                        classes_cnt[1] += 1
                        # for idx in patchInfo['clsid']:
                        #     classes_cnt[idx] += 1
        
        result_table = PrettyTable()
        result_table.field_names = ['Neg', 'Pos', 'Total']
        result_table.add_row([classes_cnt[0], classes_cnt[1], len(datalist)])
        print(result_table)
    
        with open(f'{json_root}/{mode}_patches_v0403.json', 'w') as f:
            json.dump(datalist, f)
            
'''
train:
+-------+-------+-------+
|  Neg  |  Pos  | Total |
+-------+-------+-------+
| 45305 | 53973 | 99278 |
+-------+-------+-------+

val:
+-------+-------+-------+
|  Neg  |  Pos  | Total |
+-------+-------+-------+
| 11551 | 14182 | 25733 |
+-------+-------+-------+
'''