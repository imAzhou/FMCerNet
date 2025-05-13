import json

RECORD_CLASS = {
    'NILM':'NILM',
    'GEC':'NILM',
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

def load_partial_list(roi_patientIds):
    with open(f'{root_dir}/train_partial_pos.json', 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(f'{root_dir}/val_partial_pos.json', 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    train_patchlist,val_patchlist = [],[]
    for kfbinfo in train_data:
        if kfbinfo['patientId'] in roi_patientIds:
            continue
        train_patchlist.extend(kfbinfo['patch_list'])
    for kfbinfo in val_data:
        if kfbinfo['patientId'] in roi_patientIds:
            continue
        val_patchlist.extend(kfbinfo['patch_list'])
    
    return train_patchlist,val_patchlist

def load_total_list():
    with open(f'{root_dir}/train_roi_total.json', 'r', encoding='utf-8') as f:
        train_roi_data = json.load(f)
    with open(f'{root_dir}/val_roi_total.json', 'r', encoding='utf-8') as f:
        val_roi_data = json.load(f)
    with open(f'{root_dir}/train_slide_total.json', 'r', encoding='utf-8') as f:
        train_slide_data = json.load(f)

    train_patchlist,val_patchlist = [],[]
    roi_patientIds = []
    for kfbinfo in train_roi_data:
        roi_patientIds.append(kfbinfo['patientId'])
        train_patchlist.extend(kfbinfo['patchlist'])
    for kfbinfo in val_roi_data:
        roi_patientIds.append(kfbinfo['patientId'])
        val_patchlist.extend(kfbinfo['patchlist'])
    
    for kfbinfo in train_slide_data:
        train_patchlist.extend(kfbinfo['patchlist'])
    
    return train_patchlist,val_patchlist,roi_patientIds

def load_neg_list():
    with open(f'{root_dir}/train_negslide_patches.json', 'r', encoding='utf-8') as f:
        train_neg_data = json.load(f)
    with open(f'{root_dir}/val_negslide_patches.json', 'r', encoding='utf-8') as f:
        val_neg_data = json.load(f)

    train_patchlist,val_patchlist = [],[]
    train_patchlist,val_patchlist = [],[]
    for kfbinfo in train_neg_data:
        train_patchlist.extend(kfbinfo['patch_list'])
    for kfbinfo in val_neg_data:
        val_patchlist.extend(kfbinfo['patch_list'])
    
    return train_patchlist,val_patchlist

def analyze_patchlist(total_patchlist):    
    pn_cnt = [0,0]
    for patchinfo in total_patchlist:
        pn_cnt[patchinfo['diagnose']] += 1
    print(pn_cnt)

def remap_clsname(total_patchlist):    

    for patchinfo in total_patchlist:
        patchinfo['clsnames'] = [RECORD_CLASS[clsname] for clsname in patchinfo['clsnames']]
    
    return total_patchlist

def main():
    train_total,val_total,roi_patientIds = load_total_list()
    train_partial, val_partial = load_partial_list(roi_patientIds)
    train_neg,val_neg = load_neg_list()

    train_data = [*train_partial, *train_total, *train_neg]
    val_data = [*val_partial, *val_total, *val_neg]

    train_data = remap_clsname(train_data)
    val_data = remap_clsname(val_data)

    analyze_patchlist(train_data)
    analyze_patchlist(val_data)

    with open(f'{root_dir}/train_pure.json', 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)
    with open(f'{root_dir}/val_pure.json', 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    root_dir = '/c22073/zly/datasets/CervicalDatasets/LCerScanv4/annofiles'
    main()

'''
Neg,Pos
train: [70131, 75465]
val: [17208, 19962]

pure
Neg,Pos
train: [70131, 39677]
val: [17208, 11179]

roi_cut:
train Neg,Pos: [8836, 6912]
val Neg,Pos: [2351, 1818]
slide_cut:
train Neg,Pos: [3824, 1662]
'''