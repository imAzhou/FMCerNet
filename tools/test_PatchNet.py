import torch
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=r"^xFormers is available \(.*\)")
from tqdm import tqdm
import mmengine.dist as dist
from mmengine.fileio import dump
import argparse
from mmengine.config import Config
from cerwsi.datasets import load_data
from cerwsi.nets import PatchNet,SlideNet
from cerwsi.utils import set_seed, init_distributed_mode, is_main_process


parser = argparse.ArgumentParser()
# base args
parser.add_argument('config_file', type=str)
parser.add_argument('ckpt', type=str)
parser.add_argument('save_dir', type=str)
parser.add_argument('--val_json', type=str, help='assign val jsondatas')
parser.add_argument('--save_result', action='store_true')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--print_interval', type=int, default=10, help='random seed')
parser.add_argument('--world_size', default=3, type=int, help='number of distributed processes')
parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

args = parser.parse_args()

def test_net(cfg, model):
    valloader = load_data(cfg, ['val'])

    model.eval()
    pbar = valloader
    if is_main_process():
        pbar = tqdm(valloader, ncols=80)
    
    batch_outputs = []
    for idx, data_batch in enumerate(pbar):
        # if idx > 2:
        #     break
        with torch.no_grad():
            outputs = model(data_batch, 'val')
            #推理的时候训练集和验证集都要跑一下
        model.module.taskhead.evaluator.process(data_samples=outputs, data_batch=None)
        
        if args.save_result:
            batch_outputs.extend([item.cpu() for item in outputs])
    if args.save_result:
        results = dist.collect_results(batch_outputs, len(valloader.dataset), device='cpu')
    
    metrics = model.module.taskhead.evaluator.evaluate(len(valloader.dataset))
    if is_main_process():
        pbar.close()
        print(metrics)
        if args.save_result:
            out_file_path = f'{args.save_dir}/pred_result.pkl'
            dump(results, out_file_path)
            print(f'Results saved in {out_file_path}')

def main():
    init_distributed_mode(args)
    set_seed(args.seed)
    device = torch.device(f'cuda:{os.getenv("LOCAL_RANK")}')

    cfg = Config.fromfile(args.config_file)
    cfg.save_result_dir = args.save_dir
    if cfg.net_type == 'patch':
        cfg.backbone_cfg['backbone_ckpt'] = None
        cfg.instance_ckpt = None
        if args.val_json:
            cfg.val_datasets['ann_file'] = args.val_json
        model = PatchNet(cfg).to(device)
    elif cfg.net_type == 'slide':
        model = SlideNet(cfg).to(device)

    model.load_ckpt(args.ckpt)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    test_net(cfg, model)
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()


'''
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=12347 tools/test_PatchNet.py \
    log/WS1200/ml_decoder/input_224/config.py \
    log/WS1200/ml_decoder/input_224/checkpoints/best.pth \
    log/WS1200/ml_decoder/input_224 \
    --save_result
    --val_json annofiles/multilabel_puretrain.json \
    --save_result

log/slide_mc/cell_detector/2025_10_13_05_45_43/config.py
log/slide_mc/cell_detector/2025_10_13_05_45_43/checkpoints/best.pth

log/slide_mc/ml_decoder/config.py
log/slide_mc/ml_decoder/checkpoints/best.pth

log/slide_mc/ours_WS800/2025_10_10_07_40_26/config.py
log/slide_mc/ours_WS800/2025_10_10_07_40_26/checkpoints/best.pth  
'''