import torch
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=r"^xFormers is available \(.*\)")
from tqdm import tqdm
import torch.distributed as dist
import argparse
from mmengine.config import Config
from cerwsi.datasets import load_data
from cerwsi.nets import PatchClsNet
from cerwsi.utils import set_seed, init_distributed_mode, is_main_process


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

def test_net(cfg, model, model_without_ddp):
    valloader = load_data(cfg, ['val'])

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

    # if args.distributed:
    #     dist.barrier()
    #     dist.destroy_process_group()

if __name__ == '__main__':
    main()
    # analyze(f'{args.save_dir}/pred_results_0.5.json')

'''
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun  --nproc_per_node=8 --master_port=12340 test_PatchClsNet.py \
    log/WINDOW_SIZE_512/instance/2025_06_17_10_42_02/config.py \
    log/WINDOW_SIZE_512/instance/2025_06_17_10_42_02/checkpoints/best.pth \
    log/WINDOW_SIZE_512/instance/2025_06_17_10_42_02
'''