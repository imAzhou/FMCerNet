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
from fmcernet.nets import PatchNet,SlideNet
from fmcernet.datasets import load_data
from fmcernet.utils import set_seed, get_logger, init_distributed_mode, is_main_process,build_param_scheduler, lr_scheduler_step, scale_lr


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
    # real_bs = args.world_size * cfg.train_bs
    # scale_lr(real_bs, optimizer, cfg.auto_scale_lr)
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
            val_loss_stat = torch.zeros(2, device=model.module.device)
            val_loss_dict_stats = {}
            if is_main_process():
                logger.info(f'Val Epoch {epoch + 1}/{cfg.max_epochs}')
                pbar = tqdm(valloader, ncols=80)
            
            for idx, data_batch in enumerate(pbar):
                # if idx > 2:
                #     break
                with torch.no_grad():
                    outputs, val_loss, val_loss_dict = model(data_batch, 'val')
                batch_size = len(data_batch['data_samples'])
                val_loss_stat[0] += val_loss.detach() * batch_size
                val_loss_stat[1] += batch_size
                for loss_name, loss_value in val_loss_dict.items():
                    if loss_name not in val_loss_dict_stats:
                        val_loss_dict_stats[loss_name] = torch.zeros(2, device=model.module.device)
                    if torch.is_tensor(loss_value):
                        loss_value = loss_value.detach().to(model.module.device)
                    else:
                        loss_value = torch.tensor(float(loss_value), device=model.module.device)
                    val_loss_dict_stats[loss_name][0] += loss_value * batch_size
                    val_loss_dict_stats[loss_name][1] += batch_size
                model.module.taskhead.evaluator.process(data_samples=outputs, data_batch=None)

            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(val_loss_stat)
                for loss_name in sorted(val_loss_dict_stats):
                    torch.distributed.all_reduce(val_loss_dict_stats[loss_name])
            val_loss = val_loss_stat[0] / val_loss_stat[1].clamp(min=1)
            val_loss_dict = {
                loss_name: stat[0] / stat[1].clamp(min=1)
                for loss_name, stat in val_loss_dict_stats.items()
            }
            metrics = model.module.taskhead.evaluator.evaluate(len(valloader.dataset))
            if is_main_process():
                pbar.close()
                print_str = f'Val loss: {val_loss.item():.4f}'
                for loss_name, loss_value in val_loss_dict.items():
                    print_str += f', {loss_name}:{loss_value.item():.6f}'
                logger.info(print_str)
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
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun  --nproc_per_node=8 --master_port=12340 main4PatchNet.py \
    configs/dataset/l_cerscan_ws1600.py \
    configs/model/wscernet.py \
    configs/strategy_patch.py \
    --record_save_dir work_dir/mlc/ours/ws1600
    
l_cerscanv1_dataset
cdetector_ws400
hicervix_dataset

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun  --nproc_per_node=8 --master_port=12345 main4PatchNet.py \
    configs/dataset/mmpretrain/jfsw_attri_dataset.py \
    configs/model/attri_cls.py \
    configs/strategy_patch.py \
    --record_save_dir log/attri_cls/hap_lora
'''
