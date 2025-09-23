import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=r"^xFormers is available \(.*\)")
import torch
import os
from glob import glob
from tqdm import tqdm
from mmengine.config import Config
import argparse
from cerwsi.nets import PatchNet
from cerwsi.utils import set_seed, init_distributed_mode, is_main_process

from torch.utils.data import Dataset
import json
from torch.utils.data import DataLoader
from mmengine.dataset.sampler import DefaultSampler
import cv2
from mmpretrain.structures import DataSample

valid_patchlist_jsonpath = '/nfs-medical3/zly/valid_patches_WS1600.json'
# reload_patchlist_jsonpath = '/nfs-medical3/zly/reload_patches_WS1600.json'
reload_patchlist_jsonpath = '/nfs-medical3/zly/final_valid_patches_WS1600.json'
img_rootdir = '/nfs-medical3/zly/WS1600'
SEED = 1234
PATCH_EDGE = 1600
test_bs = 64
pnmodel_rootdir = 'log/WS1600/mlc_f1_34.81'
mmcls_config_file = f'{pnmodel_rootdir}/config.py'
mmcls_ckpt = f'{pnmodel_rootdir}/checkpoints/best.pth'
WSI_feat_savedir = f'data_resource/0630/WINDOW_SIZE_{PATCH_EDGE}/slide_feat_ours'
os.makedirs(WSI_feat_savedir, exist_ok=True, mode=0o777)


class OfflineDataset(Dataset):
    def __init__(self, valid_patchlist_jsonpath, img_rootdir, inputsize):
        with open(valid_patchlist_jsonpath, 'r', encoding='utf-8') as f:
            self.patchlist = json.load(f)
        self.img_rootdir = img_rootdir
        self.inputsize = inputsize

    def __len__(self):
        return len(self.patchlist)

    def __getitem__(self, idx):
        patchinfo = self.patchlist[idx]
        pid = patchinfo['patientId']
        patchinfo['img_path'] = f'{self.img_rootdir}/{pid}/{patchinfo["filename"]}'
        img = cv2.imread(patchinfo['img_path'])
        try:
            img_input = torch.as_tensor(cv2.resize(img, (self.inputsize, self.inputsize)))
            img_input = img_input.permute(2,0,1)    # (3, h, w)
            data_samples = DataSample(metainfo=patchinfo)
            return {
                'inputs': img_input,
                'data_samples': data_samples
            }
        except Exception as e:
            print(patchinfo['img_path'])
            print(e)
        

def custom_collate(batch):
    images = [item['inputs'] for item in batch]
    data_samples = [item['data_samples'] for item in batch]
    images_tensor = torch.stack(images, dim=0)
    
    return {
        'inputs': images_tensor,
        'data_samples': data_samples,
    }

def get_dataloader(
        valid_patchlist_jsonpath, img_rootdir, inputsize,
        batch_size
):
    dataset = OfflineDataset(valid_patchlist_jsonpath, img_rootdir, inputsize)
    sampler = DefaultSampler(dataset)
    loader = DataLoader(dataset, 
            pin_memory = True,
            batch_size = batch_size, 
            sampler = sampler,
            collate_fn = custom_collate,
            num_workers = 8)
    return loader

def run_inference(mlcls_model):
    dataloader = get_dataloader(
        reload_patchlist_jsonpath, img_rootdir, mlcls_model.module.img_size,
        test_bs
    )
    pbar = dataloader
    if is_main_process():
        pbar = tqdm(dataloader, ncols=80)

    for idx, data_batch in enumerate(pbar):
        with torch.no_grad():
            outputs = mlcls_model(data_batch, 'val')
        for datasample in outputs:
            img_prob = datasample.img_prob.detach().cpu()
            img_token = datasample.img_token.detach().cpu()
            save_tensor = torch.cat([img_prob.unsqueeze(0), img_token])
            purename = datasample.filename.split('.')[0]
            savedir = f"{WSI_feat_savedir}/{datasample.patientId}"
            os.makedirs(savedir, exist_ok=True, mode=0o777)
            torch.save(save_tensor, f"{savedir}/{purename}.pt")

def get_models(device, gpu):
    cfg = Config.fromfile(mmcls_config_file)
    cfg.backbone_cfg['backbone_ckpt'] = None
    mlcls_model = PatchNet(cfg).to(device)
    mlcls_model.img_size = cfg.input_size
    mlcls_model.load_ckpt(mmcls_ckpt)
    mlcls_model.eval()
    mlcls_model = torch.nn.parallel.DistributedDataParallel(
        mlcls_model, device_ids=[gpu], find_unused_parameters=False)

    return mlcls_model

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    init_distributed_mode(args)
    set_seed(SEED)
    device = torch.device(f'cuda:{os.getenv("LOCAL_RANK")}')
    
    mlcls_model = get_models(device,args.gpu)
    run_inference(mlcls_model)
    torch.distributed.destroy_process_group()

def reload_patchlist():
    with open(valid_patchlist_jsonpath, 'r', encoding='utf-8') as f:
        total_patchlist = json.load(f)
    all_pt_files = set(glob(f"{WSI_feat_savedir}/**/*.pt", recursive=True))
    new_patchlist = []
    for patchinfo in tqdm(total_patchlist, ncols=80):
        purename = patchinfo['filename'].split('.')[0]
        savepath = f"{WSI_feat_savedir}/{patchinfo['patientId']}/{purename}.pt"
        if savepath not in all_pt_files:
            new_patchlist.append(patchinfo)
    with open(reload_patchlist_jsonpath, 'w', encoding='utf-8') as f:
        json.dump(new_patchlist, f)

    print(f"Reload patch list: {len(total_patchlist)} -> {len(new_patchlist)}")

if __name__ == "__main__":
    main()
    # reload_patchlist()

'''
9.2T
Remap done. Patch nums: 3816382, Patient nums: 5689
Reload patch list: 3816382 -> 1104056

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=12341 scripts/process_WSI/offline_gene_slidefeats.py
'''