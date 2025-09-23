import os
import json
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import torch.distributed as dist
import random
import torch
from PIL import Image
import cv2
import numpy as np
import argparse
import glob
from mmpretrain.structures import DataSample
from cerwsi.nets import ValidClsNet
from cerwsi.utils import KFBSlide,set_seed, init_distributed_mode, is_main_process

WINDOW_SIZE = 1600
POSITIVE_CLASS = ['AGC', 'ASC-US','LSIL', 'ASC-H', 'HSIL']
CLASS_COLORS = [[31,119,180], [255,153,153], [255,105,180], [255,20,147], [139,0,139]]
max_negpatch_nums = 20    # 在每张 slide 中，阴性 tile 块的数量最多是 max_negpatch_nums 张, -1 则取所有
max_try = 100   # 约束：neg_patch_thr <= max_try,  从 neg slide 中裁切 tile 块最多循环的次数
data_root = f'data_resource/0630/WINDOW_SIZE_{WINDOW_SIZE}'
inter_ann_dir = f'{data_root}/ann_jsons'
neg_slide_img_savedir = f'{data_root}/images/neg_slide'
os.makedirs(neg_slide_img_savedir, exist_ok=True, mode=0o777)

def cut_negslide():
    LEVEL = 0
    CERTAIN_THR = 0.7
    SAFE_MARGIN = 100
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    init_distributed_mode(args)
    set_seed(1234)
    device = torch.device(f'cuda:{os.getenv("LOCAL_RANK")}')
    valid_model = ValidClsNet()
    valid_model.to(device)
    valid_model.device = device
    valid_model.eval()
    valid_model.load_state_dict(torch.load('checkpoints/valid_cls_best.pth'))
    valid_model = torch.nn.parallel.DistributedDataParallel(
        valid_model, device_ids=[args.gpu], find_unused_parameters=False)
    valid_model = valid_model.module

    data_group = {
        # 'puretrain': 'data_resource/0630/4_pure_train.csv',
        'val': 'data_resource/0630/6_val.csv'
    }
    for tag,csvpath in data_group.items():
        df = pd.read_csv(csvpath)
        df = df.drop_duplicates(subset=["patientId"])   # 按 patientId 去重
        df = df[df['kfb_clsid']==0] # 留下阴性切片
        data_list = df.to_dict(orient="records")  # 每一行 -> dict

        # ---- 数据切分（保证每张卡处理的数据不重复） ----
        rank = args.rank
        world_size = args.world_size
        data_per_rank = data_list[rank::world_size]

        neg_patch_list = []
        for idx,row in enumerate(data_per_rank):
            kfb_path, patientId = row["kfb_path"], row["patientId"]
            slide = KFBSlide(kfb_path)
            max_x, max_y = slide.level_dimensions[LEVEL]
            max_x, max_y = max_x-SAFE_MARGIN, max_y-SAFE_MARGIN
            slide_patch_cnt, try_cnt = 0, 0
            while slide_patch_cnt < max_negpatch_nums and try_cnt < max_try:
                x1,y1 = random.randint(SAFE_MARGIN, max_x-WINDOW_SIZE),random.randint(SAFE_MARGIN, max_y-WINDOW_SIZE)
                read_result = Image.fromarray(slide.read_region((x1,y1), LEVEL, (WINDOW_SIZE,WINDOW_SIZE)))
                data_batch = dict(inputs=[], data_samples=[])
                img_input = cv2.cvtColor(np.array(read_result), cv2.COLOR_RGB2BGR)
                img_input = torch.as_tensor(cv2.resize(img_input, (224,224)))
                data_batch['inputs'].append(img_input.permute(2,0,1))    # (bs, 3, h, w)
                data_batch['data_samples'].append(DataSample())
                data_batch['inputs'] = torch.stack(data_batch['inputs'], dim=0)
                with torch.no_grad():
                    outputs = valid_model.val_step(data_batch)
            
                if max(outputs[0].pred_score) > CERTAIN_THR and outputs[0].pred_label == 1:
                    filename = f'{patientId}_round0_{slide_patch_cnt}.png'
                    neg_patch_list.append({
                        'patientId': patientId,
                        'filename': filename,
                        'square_coords': (x1,y1,x1+WINDOW_SIZE,y1+WINDOW_SIZE),    # 在媒体资源中的相对坐标
                        'bboxes': [],
                        'clsnames': [],
                        'prefix': 'neg_slide',
                        'diagnose': 0,
                        'maskfile': ''
                    })
                    read_result.save(f'{neg_slide_img_savedir}/{filename}')
                    slide_patch_cnt += 1
                try_cnt += 1
            
            print(f"\r[Rank {rank}] Processing {idx}/{len(data_per_rank)} samples.\t", end='')
            if(slide_patch_cnt != max_negpatch_nums):
                print(f"[Rank {rank}] slide_patch_cnt={slide_patch_cnt}, try_cnt={try_cnt}, patientId = {patientId}")
                
        print(f'Process {rank} Done, neg_patch_list nums = {len(neg_patch_list)}!', flush=True)
        all_results = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(all_results, neg_patch_list)

        if is_main_process():
            merged = []
            for r in all_results:
                merged.extend(r)
            json_savepath = f'{inter_ann_dir}/negslide_round0_{tag}.json'
            with open(json_savepath, 'w', encoding='utf-8') as f:
                json.dump(merged, f, ensure_ascii=False)
            print(f'============================')
            print(f'Total patches num : {len(merged)}; ')
            print(f'JSON file saved in : {json_savepath}')
            print(f'============================')
    torch.distributed.destroy_process_group()

def concat_patchlist():
    with open(f'{data_root}/ann_jsons/patches_in_RoI_pure_valid.json', 'r', encoding='utf-8') as f:
        RoI_patchlist = json.load(f)
    with open(f'{data_root}/ann_jsons/patches_in_RoI_jfsw_valid.json', 'r', encoding='utf-8') as f:
        jfsw_pos_patchdata = json.load(f)
    # with open(f'{data_root}/ann_jsons/patches_in_negslide_hs0.json', 'r', encoding='utf-8') as f:
    #     negslide_patchlist = json.load(f)
    
    patient2patchlist = defaultdict(list)
    for item in RoI_patchlist:
        patient2patchlist[item['patientId']].append(item)
    
    data_group = {
        'puretrain': 'data_resource/WINDOW_SIZE_1600/annofiles/4_pure_train.csv',
        'val': 'data_resource/WINDOW_SIZE_1600/annofiles/6_val.csv'
    }
    for tag,csvpath in data_group.items():
        multilabel_pn_cnt, binary_pn_cnt = [0,0],[0,0]
        df_data = pd.read_csv(csvpath)
        print(f'Load {tag} patchlist...')
        patchlist = []
        for row in tqdm(df_data.itertuples(index=False), total=len(df_data), ncols=80):
            samplelist = patient2patchlist[row.patientId]
            if tag == 'puretrain' and neg_patch_thr > 0:
                poslist = [i for i in samplelist if i['diagnose']==1]
                neglist = [i for i in samplelist if i['diagnose']==0]
                random.shuffle(neglist)
                samplelist = [*poslist, *neglist[:neg_patch_thr]]
            patchlist.extend(samplelist)
        
        if tag == 'puretrain':
            # patchlist.extend(negslide_patchlist)
            patchlist.extend(jfsw_pos_patchdata)

        multilabel_jsondata = {
            "metainfo": {"classes": POSITIVE_CLASS},
            "data_list": []
        }
        binarylabel_txtdata = []

        print(f'Format {tag} patchlist to mmcls...')
        for patchinfo in tqdm(patchlist, ncols=80):
            imgname = f"{patchinfo['prefix']}/{patchinfo['filename']}"
            if patchinfo['prefix'] == 'total_pos':
                clsids = []
                for i in patchinfo['clsnames']:
                    if i == 'SCC':
                        i = 'HSIL'
                    clsids.append(POSITIVE_CLASS.index(i))

                multilabel_jsondata['data_list'].append({
                    "img_path": imgname,
                    "gt_label": list(set(clsids))
                })
                multilabel_pn_cnt[patchinfo['diagnose']] += 1

            binarylabel_txtdata.append(f'{imgname} {patchinfo["diagnose"]}\n')
            binary_pn_cnt[patchinfo['diagnose']] += 1
        
        print(f'{tag} multilabel_pn_cnt: {multilabel_pn_cnt}')
        print(f'{tag} binary_pn_cnt: {binary_pn_cnt}')

        if tag == 'puretrain' and neg_patch_thr > 0:
            tag += f'_npt{neg_patch_thr}'
        with open(f'{ann_dir}/multilabel_{tag}_noneg.json', 'w', encoding='utf-8') as f:
            json.dump(multilabel_jsondata, f, ensure_ascii=False)
        # with open(f'{ann_dir}/binarylabel_{tag}.txt', 'w', encoding='utf-8') as f:
        #     f.writelines(binarylabel_txtdata)

def add_negslide():
    pid_patchlist = defaultdict(list)
    for filename in tqdm(os.listdir(f'{data_root}/images/neg_slide_rc'), ncols=80):
        pid = filename.split('_rc')[0]
        pid_patchlist[pid].append({
            "img_path": f"neg_slide_rc/{filename}",
            "gt_label": []
        })
        
    with open(f'{data_root}/annofiles/multilabel_puretrain.json', 'r', encoding='utf-8') as f:
        traindata = json.load(f)
    with open(f'{data_root}/annofiles/multilabel_val.json', 'r', encoding='utf-8') as f:
        valdata = json.load(f)
    df_train = pd.read_csv(f'{data_root}/annofiles/45_purejfsw_train.csv')
    df_val = pd.read_csv(f'{data_root}/annofiles/6_val.csv')
    new_datalist = []
    for row in tqdm(df_train.itertuples(index=False), total=len(df_train), ncols=80):
        new_datalist.extend(pid_patchlist[row.patientId])
    traindata['data_list'].extend(new_datalist)
    with open(f'{data_root}/annofiles/multilabel_extendtrain.json', 'w', encoding='utf-8') as f:
        json.dump(traindata, f, ensure_ascii=False)
    pn_cnt = [0,0]
    for datainfo in traindata['data_list']:
        diag = int(len(datainfo['gt_label'])>0)
        pn_cnt[diag] += 1
    print(f'Train add {len(new_datalist)} patches, pn_cnt: {pn_cnt}')

    new_datalist = []
    for row in tqdm(df_val.itertuples(index=False), total=len(df_val), ncols=80):
        new_datalist.extend(pid_patchlist[row.patientId])
    valdata['data_list'].extend(new_datalist)
    with open(f'{data_root}/annofiles/multilabel_extendval.json', 'w', encoding='utf-8') as f:
        json.dump(valdata, f, ensure_ascii=False)
    pn_cnt = [0,0]
    for datainfo in valdata['data_list']:
        diag = int(len(datainfo['gt_label'])>0)
        pn_cnt[diag] += 1
    print(f'Val add {len(new_datalist)} patches, pn_cnt: {pn_cnt}')

def extract_posslide():
    with open(f'{data_root}/ann_jsons/patches_in_RoI_pure_valid.json', 'r', encoding='utf-8') as f:
        RoI_patchlist = json.load(f)

    patient2patchlist = defaultdict(list)
    for patchInfo in RoI_patchlist:
        patient2patchlist[patchInfo['patientId']].append(patchInfo)
    
    data_group = {
        'puretrain': 'data_resource/0630/4_pure_train.csv',
        'val': 'data_resource/0630/6_val.csv'
    }
    for tag,csvpath in data_group.items():
        multilabel_pn_cnt = [0,0]
        multilabel_jsondata = {
            "metainfo": {"classes": POSITIVE_CLASS},
            "data_list": []
        }
        df_data = pd.read_csv(csvpath)
        for row in tqdm(df_data.itertuples(index=False), total=len(df_data), ncols=80):
            slide_patchlist = patient2patchlist[row.patientId]
            poslist = [item for item in slide_patchlist if item['prefix'] == 'total_pos']
            neglist = [item for item in slide_patchlist if item['diagnose'] == 0]
            random.shuffle(neglist)
            chosed_patchlist = [*poslist, *neglist[:max_negpatch_nums]]
            for patchinfo in chosed_patchlist:
                imgname = f"{patchinfo['prefix']}/{patchinfo['filename']}"
                gt_label = []
                if patchinfo['prefix'] == 'total_pos':
                    clsids = []
                    for i in patchinfo['clsnames']:
                        if i == 'SCC':
                            i = 'HSIL'
                        clsids.append(POSITIVE_CLASS.index(i))
                    gt_label = list(set(clsids))

                multilabel_jsondata['data_list'].append({
                    "img_path": imgname,
                    "gt_label": gt_label
                })
                multilabel_pn_cnt[patchinfo['diagnose']] += 1
        
        print(f'{tag} multilabel_pn_cnt: {multilabel_pn_cnt}')

        with open(f'{inter_ann_dir}/multilabel_{tag}_noneg.json', 'w', encoding='utf-8') as f:
            json.dump(multilabel_jsondata, f, ensure_ascii=False)
    
def clear_imgs():
    keep_filename = []
    for tag in ['puretrain','val']:
        with open(f'{inter_ann_dir}/negslide_round0_{tag}.json', 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        filenames = [item['filename'] for item in json_data]
        keep_filename.extend(filenames)
    
    exists_imgpath = glob.glob(f'{neg_slide_img_savedir}/*.png')
    print(f'keep_filename nums: {len(set(keep_filename))}')
    print(f'exists_imgpath nums: {len(exists_imgpath)}')
    # for imgpath in tqdm(exists_imgpath, ncols=80):
    #     filename = os.path.basename(imgpath)
    #     if filename not in keep_filename:
    #         os.remove(imgpath)
    
if __name__ == "__main__":
    ann_dir = f'{data_root}/annofiles'
    os.makedirs(ann_dir, exist_ok=True, mode=0o777)
    # extract_posslide()
    # cut_negslide()
    clear_imgs()
    # partial_pos 样本只会用于阴阳二分类,不会用于多标签分类
    # neg_patch_thr 只会作用于训练集，验证集保持不变（真实情况就是阳性 patch 远少于阴性 patch）
    # concat_patchlist()
    # add_negslide()

'''
WS = 850
neg_patch_thr = -1
puretrain multilabel_pn_cnt: [51450, 12607]
val multilabel_pn_cnt: [26437, 6584]

puretrain binary_pn_cnt: [51450, 47883]
val binary_pn_cnt: [26437, 6616]

neg_patch_thr = 5
puretrain multilabel_pn_cnt: [5481, 12607] + [5987, 0] = [11468, 12607]
puretrain binary_pn_cnt: [5481, 47883] + [5987, 0] = [11468, 47883]


WS = 1600, max_negpatch_nums = 20
puretrain multilabel_pn_cnt: [6172, 7396]
val multilabel_pn_cnt: [1531, 3818]


Train add 7996 patches, pn_cnt: [20810, 7801]
Val add 1495 patches, pn_cnt: [7709, 3916]

puretrain binary_pn_cnt: [12814, 21030]
val binary_pn_cnt: [6214, 3927]

neg_patch_thr = 5
puretrain multilabel_pn_cnt: [3665, 7801] + [5996, 0] = [9661, 7801]
puretrain binary_pn_cnt: [3665, 21030] + [5996, 0] = [9661, 21030]

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun  --nproc_per_node=8 --master_port=12342 scripts/data_process_0630/5_make_mlc_dataset.py
'''
    