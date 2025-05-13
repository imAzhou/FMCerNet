from cerwsi.utils import draw_OD
import json
import random
from PIL import Image
import os

POSITIVE_CLASS = ['AGC', 'ASC-US','LSIL', 'ASC-H', 'HSIL']
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
classes = ['NILM', 'AGC', 'ASC-US', 'LSIL', 'ASC-H', 'HSIL']
PATCH_EDGE = 224

with open(f'data_resource/0429/annofiles/partial_train_pos_{PATCH_EDGE}.json', 'r', encoding='utf-8') as f:
    partial_train_data = json.load(f)
with open(f'data_resource/0429/annofiles/partial_val_pos_{PATCH_EDGE}.json', 'r', encoding='utf-8') as f:
    partial_val_data = json.load(f)
with open(f'data_resource/0429/annofiles/0409_zheyi_slide_train_{PATCH_EDGE}.json', 'r', encoding='utf-8') as f:
    slide_train_0409 = json.load(f)
with open(f'data_resource/0429/annofiles/0422_zheyi_slide_train_{PATCH_EDGE}.json', 'r', encoding='utf-8') as f:
    slide_train_0422 = json.load(f)
with open(f'data_resource/0429/annofiles/0422_zheyi_slide_val_{PATCH_EDGE}.json', 'r', encoding='utf-8') as f:
    slide_val_0422 = json.load(f)
with open(f'data_resource/0429/annofiles/roi_train_{PATCH_EDGE}.json', 'r', encoding='utf-8') as f:
    roi_train = json.load(f)
with open(f'data_resource/0429/annofiles/roi_val_{PATCH_EDGE}.json', 'r', encoding='utf-8') as f:
    roi_val = json.load(f)

train_data = []
for patientItem in partial_train_data:
    train_data.extend(patientItem['patch_list'])
for patientItem in slide_train_0409:
    train_data.extend(patientItem['patchlist'])
for patientItem in slide_train_0422:
    train_data.extend(patientItem['patchlist'])
for patientItem in roi_train:
    train_data.extend(patientItem['patchlist'])

save_dir = 'statistic_results/0429/vis_sample'
os.makedirs(save_dir, exist_ok=True)
multi_cls_cnt = 0
for patchinfo in train_data:
    patchinfo['clsnames'] = [RECORD_CLASS[clsname] for clsname in patchinfo['clsnames']]
    clsids = list(set([classes.index(clsname) for clsname in patchinfo['clsnames'] if clsname != 'NILM']))
    if len(clsids) > 1:
        multi_cls_cnt += 1
    if random.random() < 0.01 and patchinfo["prefix"] == 'total_pos':
        imagepath = f'data_resource/0429/{PATCH_EDGE}/images/{patchinfo["prefix"]}/{patchinfo["filename"]}'
        save_path = f'{save_dir}/{patchinfo["prefix"]}_{patchinfo["filename"]}'
        img = Image.open(imagepath)
        inside_items = [
            dict(sub_class=clsname,region=bbox) for bbox,clsname in zip(patchinfo['bboxes'],patchinfo['clsnames'])]
        draw_OD(img,save_path,[0,0,PATCH_EDGE,PATCH_EDGE], inside_items, POSITIVE_CLASS)

print(f'Train multi_cls_cnt: {multi_cls_cnt}/{len(train_data)}')

val_data = []
for patientItem in partial_val_data:
    val_data.extend(patientItem['patch_list'])
for patientItem in slide_val_0422:
    val_data.extend(patientItem['patchlist'])
for patientItem in roi_val:
    val_data.extend(patientItem['patchlist'])

multi_cls_cnt = 0
for patchinfo in val_data:
    patchinfo['clsnames'] = [RECORD_CLASS[clsname] for clsname in patchinfo['clsnames']]
    clsids = list(set([classes.index(clsname) for clsname in patchinfo['clsnames'] if clsname != 'NILM']))
    if len(clsids) > 1:
        multi_cls_cnt += 1
print(f'Val multi_cls_cnt: {multi_cls_cnt}/{len(val_data)}')



'''
Train multi_cls_cnt: 2015/118645
Val multi_cls_cnt: 536/37737
'''