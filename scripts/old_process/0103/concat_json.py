import json
from tqdm import tqdm

def list_to_coo(matrix):
    coo = []
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            if value != 0:
                coo.append((i, j, int(value)))
    return coo

if __name__ == '__main__':
    
    json_root = '/nfs5/zly/codes/CerWSI/data_resource/0103/annofiles'
    for mode in ['train','val']:
        datalist = []
        for clstype in ['pos','neg']:
            with open(f'{json_root}/{mode}_{clstype}_patches.json', 'r') as f:
                pdata = json.load(f)

            for slideInfo in tqdm(pdata,ncols=80):
                for patchInfo in slideInfo['patch_list']:   # for in each slide patch list
                    coo_gtmap = list_to_coo(patchInfo['gtmap_14'])  # (row,col,value)

                    datalist.append({
                        'filename': patchInfo['filename'],
                        'diagnose': int(patchInfo['diagnose']),
                        'gtmap_14': coo_gtmap,
                        'prefix': 'Pos' if int(patchInfo['diagnose']) == 1 else 'Neg'
                    })
    
        with open(f'{json_root}/{mode}_patches.json', 'w') as f:
            json.dump(datalist, f)
            