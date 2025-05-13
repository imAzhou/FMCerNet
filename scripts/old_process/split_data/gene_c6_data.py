from tqdm import tqdm
import random
from prettytable import PrettyTable

root_dir = 'data_resource/cls_pn/cut_img'

NEGATIVE_CLASS = ['rc_NILM', 'NILM', 'GEC']
ASCUS = ['ASC-US']
LSIL = ['LSIL']
ASCH = ['ASC-H']
HSIL = ['HSIL', 'SCC']
AGC_CLASS = ['AGC-NOS', 'AGC', 'AGC-N', 'AGC-FN']
custom_order = [*NEGATIVE_CLASS, 'ASC-US', 'LSIL', 'ASC-H', 'HSIL', 'SCC', *AGC_CLASS]

for mode in ['train', 'val']:
    patch_clsname_cnt = dict()
    rc_NILM_patch_patientId = dict()
    new_lines = []
    with open(f'{root_dir}/{mode}_rcp.txt', 'r') as f:
        for line in tqdm(f.readlines(), ncols=80):
            filepath = line.split(' ')[0]
            patch_clsname = filepath.split('/')[0]
            filename = filepath.split('/')[1]
            patch_clsid = -1
            if patch_clsname in NEGATIVE_CLASS:
                patch_clsid = 0
            elif patch_clsname in ASCUS:
                patch_clsid = 1
            elif patch_clsname in LSIL:
                patch_clsid = 2
            elif patch_clsname in ASCH:
                patch_clsid = 3
            elif patch_clsname in HSIL:
                patch_clsid = 4
            elif patch_clsname in AGC_CLASS:
                patch_clsid = 5

            line_txt = f'{filepath} {patch_clsid}\n'
            if patch_clsname == 'rc_NILM':
                patientId = '_'.join(filename.split('_')[:-1])
                rc_NILM_patch_patientId.setdefault(patientId, []).append(line_txt)
            else:
                new_lines.append(line_txt)
                patch_clsname_cnt[patch_clsname] = patch_clsname_cnt.get(patch_clsname, 0) + 1
    
    for pId,linetxts in rc_NILM_patch_patientId.items():
        random.shuffle(linetxts)
        keep_idx = len(linetxts)
        new_lines.extend(linetxts[:keep_idx])
        patch_clsname_cnt['rc_NILM'] = patch_clsname_cnt.get('rc_NILM', 0) + keep_idx
    
    
    sorted_keys = [key for key in custom_order if key in patch_clsname_cnt]
    sorted_values = [patch_clsname_cnt[key] for key in sorted_keys]
    result_table = PrettyTable(title=f'{mode} Patch Nums')
    result_table.field_names = ["类别"] + sorted_keys
    result_table.add_row(['num'] + sorted_values)
    print(result_table)

    random.shuffle(new_lines)
    with open(f'{root_dir}/{mode}_rcp_c6.txt', 'w') as f:
        f.writelines(new_lines)