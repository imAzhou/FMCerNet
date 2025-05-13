import torch
import os
from tqdm import tqdm
import torch.distributed as dist
import argparse
from mmengine.config import Config
from cerwsi.datasets import load_data
from cerwsi.nets import PatchClsNet
from cerwsi.utils import set_seed, init_distributed_mode, is_main_process
import json
from PIL import Image

from prettytable import PrettyTable
from cerwsi.utils import calculate_metrics,print_confusion_matrix,draw_OD

POSITIVE_THR = 0.5
POSITIVE_CLASS = ['AGC', 'ASC-US','LSIL', 'ASC-H', 'HSIL']
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

parser = argparse.ArgumentParser()
# base args
parser.add_argument('config_file', type=str)
parser.add_argument('ckpt', type=str)
parser.add_argument('save_dir', type=str)
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--print_interval', type=int, default=10, help='random seed')
parser.add_argument('--world_size', default=3, type=int, help='number of distributed processes')
parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

args = parser.parse_args()

def draw_pred(img_item):
    img_path = img_item['img_path']
    img = Image.open(img_path)
    w,h = img.size
    filename = os.path.basename(img_path)
    square_coords = [0,0,w,h]
    inside_items = []
    scale_ratio = h//14
    for tk in img_item['token_labels']:
        row,col,clsid = tk
        y1,x1 = row*scale_ratio,col*scale_ratio
        clsname = POSITIVE_CLASS[clsid-1]
        inside_items.append(
            dict(sub_class=clsname, region=dict(x=x1,y=y1,width=scale_ratio,height=scale_ratio))
        )
    draw_OD(img, f'{args.save_dir}/FN/{filename}', square_coords, inside_items, POSITIVE_CLASS)


def test_net(cfg, model, model_without_ddp):
    trainloader,valloader = load_data(cfg)

    model.eval()
    pbar = valloader
    if is_main_process():
        pbar = tqdm(valloader, ncols=80)
    
    for idx, data_batch in enumerate(pbar):
        # if idx > 2:
        #     break
        with torch.no_grad():
            outputs = model(data_batch, 'val')
        model_without_ddp.classifier.evaluator.process(data_samples=[outputs], data_batch=None)
    
    metrics = model_without_ddp.classifier.evaluator.evaluate(len(valloader.dataset))
    if is_main_process():
        pbar.close()
        print(metrics)



def analyze(json_path):
    with open(json_path, 'r') as f:
        pred_results = json.load(f)
    
    y_true,y_pred = [],[]
    conflict_pred = 0
    error_pos_cls = [0]*len(POSITIVE_CLASS)
    for imgitem in tqdm(pred_results['results']):
        y_true.append(imgitem['gt_label'])
        y_pred.append(imgitem['pred_label'])
        if imgitem['pred_label'] == 1 and len(imgitem['pos_pred']) == 0:
            conflict_pred += 1
        if imgitem['gt_label'] == 1 and imgitem['pred_label'] == 0:
            os.makedirs(f'{args.save_dir}/FN',exist_ok=True)
            draw_pred(imgitem)
            tks = [tk[-1]-1 for tk in imgitem['token_labels']]
            for i in range(len(error_pos_cls)):
                if i in tks:
                    error_pos_cls[i] += 1
    error_cls_table = PrettyTable()
    error_cls_table.field_names = POSITIVE_CLASS
    error_cls_table.add_row(error_pos_cls)
    print(error_cls_table)

    metric_result = calculate_metrics(y_true,y_pred)
    cm = metric_result['cm']
    del metric_result['cm']
    result_table = PrettyTable()
    result_table.field_names = metric_result.keys()
    result_table.add_row(metric_result.values())
    print(result_table)

    print_confusion_matrix(cm)
    print(f'conflict_pred: {conflict_pred}')

def main():
    init_distributed_mode(args)
    set_seed(args.seed)
    device = torch.device(f'cuda:{os.getenv("LOCAL_RANK")}')

    cfg = Config.fromfile(args.config_file)
    cfg.save_result_dir = args.save_dir
    cfg.backbone_cfg['backbone_ckpt'] = None
    cfg.instance_ckpt = None
    model = PatchClsNet(cfg).to(device)
    model_without_ddp = model

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    model_without_ddp.load_ckpt(args.ckpt)
    test_net(cfg, model, model_without_ddp)

    if args.distributed:
        dist.destroy_process_group()

if __name__ == '__main__':
    main()
    # analyze(f'{args.save_dir}/pred_results_0.5.json')

'''
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun  --nproc_per_node=8 --master_port=12341 test_PatchClsNet.py \
    log/l_cerscanv1/wscer_partial/2025_05_12_14_52_56/config.py \
    log/l_cerscanv1/wscer_partial/2025_05_12_14_52_56/epoch_39.pth \
    log/l_cerscanv1/wscer_partial/2025_05_12_14_52_56
'''