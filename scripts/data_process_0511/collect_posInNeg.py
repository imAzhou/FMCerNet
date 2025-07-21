import glob
import json
import os
import shutil
from pycocotools.coco import COCO
from tqdm import tqdm
import random

WINDOW_SIZE = 512

def main():
    MAXKEEP = 20
    neg_patches = []
    for pId in tqdm(os.listdir(srcroot_imgdir), ncols=80):
        img_cnt = 0
        total_imgs = glob.glob(f'{srcroot_imgdir}/{pId}/1/*.png')
        random.shuffle(total_imgs)
        for imgpath in total_imgs[:MAXKEEP]:
            filename = f'pos_in_neg/{pId}_posinneg_{img_cnt}.png'
            shutil.copy(imgpath, f'{dataset_rootdir}/images/{filename}' )
            neg_patches.append({
                'width': WINDOW_SIZE,
                'height': WINDOW_SIZE,
                'file_name': filename,
                'extra_info':{'prefix': 'pos_in_neg', 'square_coords': [-1]*4},
                'diagnose': 0
            })
            img_cnt += 1
    print(f'Add {len(neg_patches)} neg patches.')

    origin_jsonfiles = [
        'puretrain_cocoformat',
        'puretrain_aug_cocoformat',
    ]
    dest_jsonfiles = [
        'puretrain_withneg_cocoformat',
        'puretrain_aug_withneg_cocoformat',
    ]
    for filename,newfilename in zip(origin_jsonfiles, dest_jsonfiles):
        coco = COCO(f'{dataset_rootdir}/annofiles/{filename}.json')
        img_id_counter = max(coco.imgs.keys()) + 1
        with open(f'{dataset_rootdir}/annofiles/{filename}.json', 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        for item in neg_patches:
            item['id'] = img_id_counter
            json_data['images'].append(item)
            img_id_counter += 1
        
        with open(f'{dataset_rootdir}/annofiles/{newfilename}.json', 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False)
    
    
    

if __name__ == "__main__":
    dataset_rootdir = f'data_resource/0511/WINDOW_SIZE_{WINDOW_SIZE}'
    srcroot_imgdir = 'log/WINDOW_SIZE_512/mAP_30.6/posInNeg'
    os.makedirs(f'{dataset_rootdir}/images/pos_in_neg', exist_ok=True, mode=0o777)
    main()