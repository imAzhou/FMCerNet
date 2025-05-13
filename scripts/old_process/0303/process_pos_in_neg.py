from tqdm import tqdm
import pandas as pd
import shutil
import json
import glob
import os

def gene_items(mode):
    save_dir = f'data_resource/0103/images/hardPos_in_NegSlide'
    os.makedirs(save_dir, exist_ok=True)

    all_pIds = os.listdir(patches_dir)
    slide_cnt,totoal_img_cnt = 0,0
    total_items = []
    for pId in tqdm(all_pIds, ncols=80):
        patient_row = df_data.loc[df_data['patientId'] == pId].iloc[0]
        slide_item = {
            'patientId': pId,
            'kfb_path': patient_row.kfb_path,
            'patch_list': []
        }
        for idx,img_path in enumerate(glob.glob(f'{patches_dir}/{pId}/1/*.png')):
            filename = f'{pId}_{idx}.png'
            slide_item['patch_list'].append({
                'filename': filename,
                'square_x1y1': [-1,-1],
                'bboxes': [],
                'clsnames': [],
                'diagnose': 0,
                'gtmap_14': []
            })
            shutil.copy(img_path,f'{save_dir}/{filename}')
        if len(slide_item['patch_list']) > 0:
            slide_cnt += 1
            totoal_img_cnt += len(slide_item['patch_list'])
            total_items.append(slide_item)
    
    print(f'mode {mode} slide: {slide_cnt}')
    print(f'mode {mode} patch: {totoal_img_cnt}')
    return total_items

if __name__ == '__main__':
    mode = 'val'
    patches_dir = f'predict_results_{mode}'
    anno_jsonpath = f'data_resource/0103/annofiles/hardNegSample_in_{mode}.json'
    df_data = pd.read_csv(f'data_resource/0103/annofiles/1223_{mode}.csv')
    total_items = gene_items(mode)
    with open(anno_jsonpath, 'w') as f:
        json.dump(total_items, f)

'''
mode train slide: 563/765, patch: 7636
mode val slide: 146/192, patch: 8625
'''