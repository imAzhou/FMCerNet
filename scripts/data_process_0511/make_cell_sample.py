import json
from tqdm import tqdm
import os
from PIL import Image
from cerwsi.utils import KFBSlide

def save_images():
    save_dir = 'data_resource/cervical_cell_seg/images'
    os.makedirs(save_dir, exist_ok=True)

    with open('data_resource/0511/ann_jsons/zheyi_roi.json', 'r', encoding='utf-8') as f:
        roi_data = json.load(f)
    with open('data_resource/0511/ann_jsons/zheyi_slide.json', 'r', encoding='utf-8') as f:
        slide_data = json.load(f)
    
    total_data = [*roi_data, *slide_data]
    cnt = [[0,0],[0,0]]
    for pInfo in tqdm(total_data, ncols=80):
        idx = 0 if pInfo['media_type'] == 'roi' else 1
        for roiinfo in pInfo['annotations']:
            rx1,ry1,rx2,ry2 = roiinfo['region']
            w,h = rx2-rx1, ry2-ry1
            if w<2048 and h<2048:
                cidx = 0 if len(roiinfo['children']) > 0 else 1
                cnt[idx][cidx] += 1
                # if roiinfo["annid"] == 1979273155785:
                #     print()
                if pInfo['media_type'] == 'roi':
                    read_img = Image.open(pInfo['source_path'])
                elif pInfo['media_type'] == 'slide':
                    slide = KFBSlide(pInfo['source_path'])
                    read_img = Image.fromarray(slide.read_region((rx1,ry1), 0, (w,h)))
                read_img.save(f'{save_dir}/{pInfo["patientId"]}_{roiinfo["annid"]}.png')
    
    print(cnt)  # [[0, 0], [77, 0]]

if __name__ == "__main__":
    save_images()