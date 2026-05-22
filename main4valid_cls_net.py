import torch
import time
from PIL import Image
from tqdm import tqdm
from collections import OrderedDict
import glob
import shutil
import os
import cv2
import numpy as np
from math import ceil
import argparse
from mmengine.config import Config
from fmcernet.nets import ValidClsNet
from fmcernet.utils import (KFBSlide, set_seed,
                          load_cls_dataset,get_logger,get_train_strategy,build_evaluator)
from mmpretrain.structures import DataSample
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.conv")

def train_net(cfg):
    model.train()

    trainloader,valloader = load_cls_dataset(d_config = cfg, seed = args.seed)
    optimizer,lr_scheduler = get_train_strategy(model, cfg)
    evaluator = build_evaluator(cfg.val_evaluator)

    logger, files_save_dir = get_logger(
        args.record_save_dir, model, cfg, 'valid_cls')
    
    for epoch in range(cfg.max_epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        pbar = tqdm(total=len(trainloader)*cfg.train_bs, desc=f'Train Epoch {epoch + 1}/{cfg.max_epochs}, LR: {current_lr:.6f}')
        for idx, data_batch in enumerate(trainloader):
            loss = model.train_step(data_batch, optim_wrapper=optimizer)
            postfix = OrderedDict()
            for key, value in loss.items():
                postfix[key] = value.item()
            pbar.set_postfix(postfix)
            pbar.update(cfg.train_bs)
        pbar.close()
        lr_scheduler.step()

        model.eval()
        logger.info(f'Val Epoch {epoch + 1}/{cfg.max_epochs}')
        pbar = tqdm(total=len(valloader)*cfg.val_bs, desc=f'Val Epoch {epoch + 1}/{cfg.max_epochs}')
        for idx, data_batch in enumerate(valloader):
            with torch.no_grad():
                outputs = model.val_step(data_batch)
            evaluator.process(data_samples=outputs, data_batch=data_batch)
            pbar.update(cfg.val_bs)

        metrics = evaluator.evaluate(len(valloader)*cfg.val_bs)            
        pbar.close()
        logger.info(metrics)

    torch.save(model.state_dict(), f'checkpoints/vlaid_cls_best.pth')

def patch_test(test_transform):
    save_root_dir = '/medical-data/zly/cervical/wxl/Pred_NILM'
    # test_img_list = glob.glob('/medical-data/zly/cervical/wxl/random_cut_NILM/*.png')
    test_img_list = glob.glob('/medical-data/zly/cervical/wxl/side_del/01S028/*.png')
    valid_cnt, invalid_cnt = 0,0
    for img_path in tqdm(test_img_list, ncols=80):
        img_tensor = test_transform(Image.open(img_path))
        img_tensor = img_tensor.unsqueeze(0).to(device)
        preds = model(img_tensor)
        pred_idx = torch.softmax(preds, dim=1)
        pred_idx = pred_idx.argmax(dim=1).tolist()
        basename = os.path.basename(img_path)
        if pred_idx[0] == 1:
            os.makedirs(f'{save_root_dir}/valid', exist_ok=True)
            shutil.copy(img_path, f'{save_root_dir}/valid/{basename}')
            valid_cnt += 1
        else:
            os.makedirs(f'{save_root_dir}/invalid', exist_ok=True)
            shutil.copy(img_path, f'{save_root_dir}/invalid/{basename}')
            invalid_cnt += 1
    
    print(f'valid_cnt: {valid_cnt}, invalid_cnt: {invalid_cnt}')


def slide_test():
    PATCH_EDGE = 500
    CERTAIN_THR = 0.7
    test_save_dir = 'predict_results'
    kfb_path = '/disk/medical_datasets/cervix/ZJU-TCT/第一批标注2023.9.5之前/NILM/01S001.kfb'
    # valid: 5797, invalid: 1339, uncertain: 170, total: 7306
    slide = KFBSlide(kfb_path)
    width, height = slide.level_dimensions[0]
    iw, ih = ceil(width/PATCH_EDGE), ceil(height/PATCH_EDGE)
    r2 = (int(max(iw, ih)*1.1)//2)**2
    cix, ciy = iw // 2, ih // 2
    patientId = os.path.basename(kfb_path).split('.')[0]
    save_root_dir = f'{test_save_dir}/w{PATCH_EDGE}/{patientId}'
    for tag in ['invalid', 'valid', 'uncertain']:
        save_dir = f'{save_root_dir}/{tag}'
        os.makedirs(save_dir, exist_ok=True)
    
    valid_cnt, invalid_cnt, uncertain_cnt = 0,0,0
    data_batch = dict(inputs=[], data_samples=[])
    read_result_pool = []
    start_time = time.time()
    for j, y in enumerate(range(0, height, PATCH_EDGE)):
        for i, x in enumerate(range(0, width, PATCH_EDGE)):
            if (i-cix)**2 + (j-ciy)**2 > r2:
                continue

            location, level, size = (x, y), 0, (PATCH_EDGE, PATCH_EDGE)
            read_result = Image.fromarray(slide.read_region(location, level, size))
            read_result_pool.append(read_result)

            img_input = cv2.cvtColor(np.array(read_result), cv2.COLOR_RGB2BGR)
            img_input = torch.as_tensor(cv2.resize(img_input, (224,224)))
            data_batch['inputs'].append(img_input.permute(2,0,1))    # (bs, 3, h, w)
            data_batch['data_samples'].append(DataSample())
            
            if len(data_batch['inputs']) % args.test_bs == 0:
                data_batch['inputs'] = torch.stack(data_batch['inputs'], dim=0)
                with torch.no_grad():
                    outputs = model.val_step(data_batch)
                for o_img,pred_output in zip(read_result_pool, outputs):
                    curent_id = valid_cnt + invalid_cnt + uncertain_cnt
                    print(f'\r当前已处理: {curent_id}', end='')

                    if max(pred_output.pred_score) > CERTAIN_THR:
                        if pred_output.pred_label == 1:
                            o_img.save(f'{save_root_dir}/valid/{patientId}_v{curent_id}.png')
                            valid_cnt += 1
                        else:
                            o_img.save(f'{save_root_dir}/invalid/{patientId}_inv{curent_id}.png')
                            invalid_cnt += 1
                    else:
                        o_img.save(f'{save_root_dir}/uncertain/{patientId}_unc{curent_id}.png')
                        uncertain_cnt += 1

                data_batch = dict(inputs=[], data_samples=[])
                read_result_pool = []
    if len(data_batch['inputs']) > 0:
        data_batch['inputs'] = torch.stack(data_batch['inputs'], dim=0)
        with torch.no_grad():
            outputs = model.val_step(data_batch)
        for o_img,pred_output in zip(read_result_pool, outputs):
            curent_id = valid_cnt + invalid_cnt + uncertain_cnt
            print(f'\r当前已处理: {curent_id}', end='')

            if max(pred_output.pred_score) > CERTAIN_THR:
                if pred_output.pred_label == 1:
                    o_img.save(f'{save_root_dir}/valid/{patientId}_v{curent_id}.png')
                    valid_cnt += 1
                else:
                    o_img.save(f'{save_root_dir}/invalid/{patientId}_inv{curent_id}.png')
                    invalid_cnt += 1
            else:
                o_img.save(f'{save_root_dir}/uncertain/{patientId}_unc{curent_id}.png')
                uncertain_cnt += 1

        data_batch = dict(inputs=[], data_samples=[])
        read_result_pool = []
    t_delta = time.time() - start_time
    print(f'\nTime of process kfb elapsed: {t_delta:0.2f} seconds, valid: {valid_cnt}, invalid: {invalid_cnt}, uncertain: {uncertain_cnt}, total: {curent_id+1}')

def test_net(net_ckpt):

    model.eval()
    model.load_state_dict(torch.load(net_ckpt))
    # patch_test(test_transform)
    slide_test()

parser = argparse.ArgumentParser()
# base args
parser.add_argument('dataset_config_file', type=str)
parser.add_argument('strategy_config_file', type=str)
parser.add_argument('--record_save_dir', type=str)
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--test_bs', type=int, default=64, help='batch size of model test')
parser.add_argument('--print_interval', type=int, default=10, help='random seed')
parser.add_argument('--device', type=str, default='cuda:0')

args = parser.parse_args()

if __name__ == '__main__':
    set_seed(args.seed)
    device = torch.device(args.device)
    d_cfg = Config.fromfile(args.dataset_config_file)
    s_cfg = Config.fromfile(args.strategy_config_file)

    cfg = Config()
    for sub_cfg in [d_cfg, s_cfg]:
        cfg.merge_from_dict(sub_cfg.to_dict())
    
    model = ValidClsNet()
    model.to(device)
    
    train_net(cfg)
    # net_ckpt = 'checkpoints/vlaid_cls_best.pth'
    # test_net(net_ckpt)

'''
python main4valid_cls_net.py \
    configs/dataset/cls_valid_dataset.py \
    configs/train_strategy.py \
    --record_save_dir log/valid_cls \
    --test_bs 32
'''