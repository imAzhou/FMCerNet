import torch
import os
import warnings
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
warnings.filterwarnings("ignore", category=UserWarning)
from tqdm import tqdm
import torch.distributed as dist
import argparse
from math import ceil
from mmengine.config import Config
from mmengine.registry import init_default_scope
from cerwsi.nets import PatchNet,ValidClsNet
from cerwsi.utils import KFBSlide,set_seed, init_distributed_mode, is_main_process
import pandas as pd
import copy
from PIL import Image
import numpy as np
from mmpretrain.structures import DataSample
import cv2

LEVEL = 0
PATCH_EDGE = 850
CERTAIN_THR = 0.7
SEED = 1234
SAFE_MARGIN = 100
infer_csv_file = 'data_resource/0630/4_pure_train.csv'
valid_ckpt = 'checkpoints/valid_cls_best.pth'
mmcls_config_file = 'log/WS850/hs_round0/config.py'
mmcls_ckpt = 'log/WS850/hs_round0/checkpoints/best.pth'
test_bs = 64

def inference_valid_batch(valid_model, read_result_pool):
    data_batch = dict(inputs=[], data_samples=[])
    for item in read_result_pool:
        img_input = torch.as_tensor(cv2.resize(item['image'], (224,224)))
        data_batch['inputs'].append(img_input.permute(2,0,1))    # (bs, 3, h, w)
        data_batch['data_samples'].append(DataSample())

    data_batch['inputs'] = torch.stack(data_batch['inputs'], dim=0)
    with torch.no_grad():
        outputs = valid_model.val_step(data_batch)
    valid_results = []  # 0: invalid   1:uncertain  2:valid
    for idx,pred_output in enumerate(outputs):
        valid_flag = 1
        if max(pred_output.pred_score) > CERTAIN_THR:
            valid_flag = 2 if pred_output.pred_label == 1 else 0
        valid_results.append(valid_flag)
    return valid_results

def inference_batch_pn(mlcls_model, valid_input):
    inputsize = mlcls_model.img_size
    data_batch = dict(inputs=[], data_samples=[])
    for item in valid_input:
        img_input = torch.as_tensor(cv2.resize(item['image'], (inputsize,inputsize)))
        data_batch['inputs'].append(img_input.permute(2,0,1))    # (bs, 3, h, w)
        data_batch['data_samples'].append(DataSample())

    data_batch['inputs'] = torch.stack(data_batch['inputs'], dim=0)
    with torch.no_grad():
        outputs = mlcls_model(data_batch, 'val')
    pred_result = []
    for bidx in range(len(img_inputs)):
        pcoords = valid_input[bidx]['coords']
        predresult = outputs[bidx].pred_instances

        pred_clsid = int(len(format_pred) > 0)
        patch_predinfo = {
            'pred_label': pred_clsid,
            'patch_coords': pcoords,
            'pos_bboxes': format_pred
        }
        pred_result.append(patch_predinfo)
        if visual_pred and str(pred_clsid) in visual_pred:
            o_img = valid_input[bidx]['image']
            timestamp = time.time()
            os.makedirs(f'{save_prefix}/{pred_clsid}', exist_ok=True)
            o_img.save(f'{save_prefix}/{pred_clsid}/{timestamp}.png')

    return pred_result

def run_inference(valid_model, mlcls_model, device):
    df = pd.read_csv(infer_csv_file)
    df = df.drop_duplicates(subset=["patientId"])   # 按 patientId 去重
    data_list = df.to_dict(orient="records")  # 每一行 -> dict
    # ---- 数据切分（保证每张卡处理的数据不重复） ----
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    data_per_rank = data_list[rank::world_size]

    pbar = data_per_rank
    if is_main_process():
        pbar = tqdm(data_per_rank, ncols=80)
    
    results = []
    for row in pbar:
        slide = KFBSlide(row["kfb_path"])
        width, height = slide.level_dimensions[LEVEL]
        width -= SAFE_MARGIN
        height -= SAFE_MARGIN
        iw, ih = ceil(width/PATCH_EDGE), ceil(height/PATCH_EDGE)
        r2 = (int(max(iw, ih)*1.1)//2)**2
        cix, ciy = iw // 2, ih // 2
        
        valid_datapool, mlcls_datapool = [],[]
        start_points = []
        for j, y in enumerate(range(0, height, PATCH_EDGE)):
            for i, x in enumerate(range(0, width, PATCH_EDGE)):
                if (i-cix)**2 + (j-ciy)**2 > r2:
                    continue
                start_points.append((x, y))
        
        for p_idx,(x,y) in enumerate(start_points):
            location, level, size = (x, y), LEVEL, (PATCH_EDGE, PATCH_EDGE)
            read_result = copy.deepcopy(Image.fromarray(slide.read_region(location, level, size)))
            coords = np.array([x, y, x+PATCH_EDGE, y+PATCH_EDGE])
            img_input = cv2.cvtColor(np.array(read_result), cv2.COLOR_RGB2BGR)
            valid_datapool.append({'image': img_input,'coords': coords.tolist()})

            if len(valid_datapool) % test_bs == 0 or p_idx == len(start_points)-1:
                valid_results = inference_valid_batch(valid_model, valid_datapool)
                mlcls_datapool.extend(
                    [valid_datapool[idx] for idx,flag in enumerate(valid_results) if flag==2])
                valid_datapool = []
            if len(mlcls_datapool) > 0:
                inference_batch_pn(mlcls_model, mlcls_datapool)


    if is_main_process():
        pbar.close()

    return results

def get_models(device, gpu):
    # init_default_scope('mmdet')
    # cfg = Config.fromfile(config_dict['detector'])
    # del cfg.model.type
    # pn_model = DINO(**cfg.model).to(device)
    # pn_model.img_size = cfg.input_size
    # pn_model.device = device
    # ckpt = torch.load(pn_model_ckpt, weights_only=False, map_location=device)['state_dict']
    # pn_model.load_state_dict(ckpt)
    # pn_model.eval()

    init_default_scope('mmpretrain')
    valid_model = ValidClsNet()
    valid_model.to(device)
    valid_model.device = device
    valid_model.eval()
    valid_model.load_state_dict(torch.load(valid_ckpt))
    valid_model = torch.nn.parallel.DistributedDataParallel(
        valid_model, device_ids=[gpu], find_unused_parameters=False)
    valid_model = valid_model.module

    cfg = Config.fromfile(mmcls_config_file)
    cfg.backbone_cfg['backbone_ckpt'] = None
    mlcls_model = PatchNet(cfg).to(device)
    mlcls_model.load_ckpt(mmcls_ckpt)
    mlcls_model.eval()
    mlcls_model = torch.nn.parallel.DistributedDataParallel(
        mlcls_model, device_ids=[gpu], find_unused_parameters=False)
    mlcls_model = mlcls_model.module

    return valid_model,mlcls_model

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    init_distributed_mode(args)
    set_seed(SEED)
    device = torch.device(f'cuda:{os.getenv("LOCAL_RANK")}')
    valid_model,mlcls_model = get_models(device,args.gpu)
    
    results = run_inference(valid_model, mlcls_model, device)
    all_results = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(all_results, results)

    if dist.get_rank() == 0:    # rank0 汇总结果
        merged = []
        for r in all_results:
            merged.extend(r)
        print(len(merged))

    torch.distributed.destroy_process_group()

if __name__ == '__main__':
    main()


'''
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=12340 tools/infer_WSI.py
'''