import torch
import os
import warnings
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
warnings.filterwarnings("ignore", category=UserWarning)
import time
from tqdm import tqdm
import argparse
from mmengine.config import Config
import torchvision
from mmengine.optim import build_optim_wrapper
torchvision.disable_beta_transforms_warning()
from cerwsi.nets import PatchNet,SlideNet
from cerwsi.datasets import load_data
from cerwsi.utils import set_seed, init_distributed_mode, get_logger, is_main_process,build_param_scheduler, lr_scheduler_step, scale_lr


parser = argparse.ArgumentParser()
# base args
parser.add_argument('dataset_config_file', type=str)
parser.add_argument('model_config_file', type=str)
parser.add_argument('strategy_config_file', type=str)
parser.add_argument('--record_save_dir', type=str)
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--world_size', default=3, type=int, help='number of distributed processes')
parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

args = parser.parse_args()


def train_net(cfg, args, model):
    trainloader,valloader = load_data(cfg, ['train','val'])
    optimizer = build_optim_wrapper(model, cfg.optim_wrapper)
    real_bs = args.world_size * cfg.train_bs
    scale_lr(real_bs, optimizer, cfg.auto_scale_lr)
    param_schedulers = build_param_scheduler(optimizer, cfg.param_scheduler, cfg.max_epochs, len(trainloader))
    if is_main_process():
        logger, files_save_dir = get_logger(args.record_save_dir, model.module, cfg)
    
    max_acc = -1
    for epoch in range(cfg.max_epochs):
        trainloader.sampler.set_epoch(epoch)
        model.train()
        current_lr = optimizer.param_groups[0]["lr"]
        pbar = trainloader
        if is_main_process():
            start_time = time.time()
            pbar = tqdm(trainloader, ncols=80)
        
        for idx, data_batch in enumerate(pbar):
            # if idx > 2:
            #     break
            loss,loss_dict = model(data_batch, 'train', optim_wrapper=optimizer)
            lr_scheduler_step(param_schedulers, 'iter')
            if is_main_process():
                pbar.desc = f"loss: {round(loss.item(), 4)}"

            if idx % 50 == 0 and is_main_process():
                print_str = f'Train Epoch [{epoch + 1}/{cfg.max_epochs}][{idx}/{len(trainloader)}], LR: {current_lr:.6f}, loss: {loss.item():.4f}'
                if len(loss_dict.keys()) > 1:
                    for k,v in loss_dict.items():
                        print_str += f', {k}:{v:.6f}'
                logger.info(print_str)
        if is_main_process():
            pbar.close()
            end_time = time.time()
            during_time = end_time - start_time
            eta_time = during_time * (cfg.max_epochs - epoch - 1)
            m, s = divmod(eta_time, 60)
            h, m = divmod(m, 60)
            print('ETA: ' + "%02d:%02d:%02d" % (h, m, s))
        
        lr_scheduler_step(param_schedulers, 'epoch')
        # if epoch > 1000:
        if (epoch+1) % cfg.val_interval == 0 or epoch == 0:
            model.eval()
            pbar = valloader
            if is_main_process():
                logger.info(f'Val Epoch {epoch + 1}/{cfg.max_epochs}')
                pbar = tqdm(valloader, ncols=80)
            
            for idx, data_batch in enumerate(pbar):
                # if idx > 2:
                #     break
                with torch.no_grad():
                    outputs = model(data_batch, 'val')
                model.module.taskhead.evaluator.process(data_samples=outputs, data_batch=None)

            metrics = model.module.taskhead.evaluator.evaluate(len(valloader.dataset))
            if is_main_process():
                pbar.close()
                # print(metrics)
                if cfg.save_each_epoch:
                    torch.save(model.module.state_dict(), f'{files_save_dir}/checkpoints/epoch_{epoch}.pth')
                prime_score_type = cfg.eval_prime_score
                prime_score = metrics[prime_score_type]
                if prime_score > max_acc:
                    max_acc = prime_score
                    print(f'Best score update: {prime_score}.')
                    torch.save(model.module.state_dict(), f'{files_save_dir}/checkpoints/best.pth')

def get_net(cfg):
    model = None
    if cfg.net_type == 'patch':
        model = PatchNet(cfg)
    elif cfg.net_type == 'slide':
        model = SlideNet(cfg)
    return model

def main():
    init_distributed_mode(args)
    set_seed(args.seed)
    device = torch.device(f'cuda:{os.getenv("LOCAL_RANK")}')

    d_cfg = Config.fromfile(args.dataset_config_file)
    m_cfg = Config.fromfile(args.model_config_file)
    s_cfg = Config.fromfile(args.strategy_config_file)

    cfg = Config()
    for sub_cfg in [d_cfg, m_cfg, s_cfg]:
        cfg.merge_from_dict(sub_cfg.to_dict())
    cfg.save_result_dir = None
    model = get_net(cfg).to(device)
    if cfg.load_from is not None:
        model.load_ckpt(cfg.load_from)
    model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True)

    train_net(cfg, args, model)
    torch.distributed.destroy_process_group()

if __name__ == '__main__':
    main()

'''
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun  --nproc_per_node=8 --master_port=12348 main4PatchNet.py \
    configs/dataset/mmpretrain/cdetector_ws400.py \
    configs/model/wscernet.py \
    configs/strategy_patch.py \
    --record_save_dir log/cdetector/wscernet
    
l_cerscanv1_dataset
cdetector_ws400
hicervix_dataset

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun  --nproc_per_node=8 --master_port=12346 main4PatchNet.py \
    configs/dataset/slide_cfg.py \
    configs/model/wsi_slidenet.py \
    configs/strategy_slide.py \
    --record_save_dir log/slide_mc/ours_WS1600
'''