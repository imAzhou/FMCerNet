import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=r"^xFormers is available \(.*\)")
import os
import time
from tqdm import tqdm
import argparse
from mmengine.config import Config
import torchvision
from mmengine.optim import build_optim_wrapper
torchvision.disable_beta_transforms_warning()
from cerwsi.nets import PatchNet,InferSegNet
from cerwsi.datasets import load_data
from cerwsi.utils import set_seed, get_logger, build_param_scheduler, lr_scheduler_step, scale_lr


parser = argparse.ArgumentParser()
# base args
parser.add_argument('dataset_config_file', type=str)
parser.add_argument('model_config_file', type=str)
parser.add_argument('strategy_config_file', type=str)
parser.add_argument('--model_tag', default='patchnet', type=str)
parser.add_argument('--record_save_dir', type=str)
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--print_interval', type=int, default=10, help='random seed')

args = parser.parse_args()


def train_net(cfg, args, model):
    trainloader,valloader = load_data(cfg, ['train','val'])
    optimizer = build_optim_wrapper(model, cfg.optim_wrapper)
    real_bs = cfg.train_bs
    scale_lr(real_bs, optimizer, cfg.auto_scale_lr)
    param_schedulers = build_param_scheduler(optimizer, cfg.param_scheduler, 
                                         cfg.max_epochs, len(trainloader))
    
    logger, files_save_dir = get_logger(args.record_save_dir, model, cfg)
    max_acc = -1
    for epoch in range(cfg.max_epochs):

        model.train()
        current_lr = optimizer.param_groups[0]["lr"]
        start_time = time.time()
        pbar = tqdm(trainloader, ncols=80)
        
        for idx, data_batch in enumerate(pbar):
            # if idx > 4:
            #     break
            loss,loss_dict = model(data_batch, 'train', optim_wrapper=optimizer)
            lr_scheduler_step(param_schedulers, 'iter')
            pbar.desc = f"loss: {round(loss.item(), 4)}"

            if idx % 50 == 0:
                print_str = f'Train Epoch [{epoch + 1}/{cfg.max_epochs}][{idx}/{len(trainloader)}], LR: {current_lr:.6f}'
                if len(loss_dict.keys()) > 1:
                    for k,v in loss_dict.items():
                        print_str += f', {k}:{v:.6f}'
                logger.info(print_str)

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
            logger.info(f'Val Epoch {epoch + 1}/{cfg.max_epochs}')
            pbar = tqdm(valloader, ncols=80)
            
            for idx, data_batch in enumerate(pbar):
                # if idx > 2:
                #     break
                with torch.no_grad():
                    outputs = model(data_batch, 'val')
                model.taskhead.evaluator.process(data_samples=outputs, data_batch=None)

            metrics = model.taskhead.evaluator.evaluate(len(valloader.dataset))

            pbar.close()
            print(metrics)
            if cfg.save_each_epoch:
                torch.save(model.state_dict(), f'{files_save_dir}/checkpoints/epoch_{epoch}.pth')
            prime_score_type = cfg.eval_prime_score
            prime_score = metrics[prime_score_type]
            if prime_score > max_acc:
                max_acc = prime_score
                print(f'Best score update: {prime_score}.')
                torch.save(model.state_dict(), f'{files_save_dir}/checkpoints/best.pth')

def main():

    set_seed(args.seed)
    
    device = torch.device(f'cuda:1')

    d_cfg = Config.fromfile(args.dataset_config_file)
    m_cfg = Config.fromfile(args.model_config_file)
    s_cfg = Config.fromfile(args.strategy_config_file)

    cfg = Config()
    for sub_cfg in [d_cfg, m_cfg, s_cfg]:
        cfg.merge_from_dict(sub_cfg.to_dict())
    cfg.save_result_dir = None
    if args.model_tag == 'patchnet':
        model = PatchNet(cfg).to(device)
    elif args.model_tag == 'inferseg':
        model = InferSegNet(cfg).to(device)
    
    if cfg.load_from is not None:
        model.load_ckpt(cfg.load_from)
    train_net(cfg, args, model)

if __name__ == '__main__':
    main()

'''
python main.py \
    configs/dataset/mmdet/hmchh_dataset.py \
    configs/model/detr.py \
    configs/strategy.py \
    --record_save_dir log/debug
'''