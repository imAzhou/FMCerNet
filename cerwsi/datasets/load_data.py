import torch
from torch.utils.data import DataLoader
from mmengine.registry import init_default_scope
from mmengine.dataset.sampler import DefaultSampler
from .cls_dataset import ClsDataset
from .instance_dataset import InstanceDataset
from mmpretrain.datasets import MultiLabelDataset

def load_data(cfg, load_modes = []):
    valid_modes = {'train', 'val', 'test'}

    assert all(mode in valid_modes for mode in load_modes), \
    f"Invalid mode(s) in load_modes: {load_modes}. Must be in {valid_modes}"

    dataloaders = []
    for mode in load_modes:
        if cfg.dataset_type == 'cls':
            init_default_scope('mmpretrain')
            annojson,transform,batch_size = get_mode_cfg(mode, cfg)
            dataset = ClsDataset(cfg, annojson, transform)
        elif cfg.dataset_type == 'multicls':
            init_default_scope('mmpretrain')
            dataset_cfg = {}
            if mode == 'train':
                dataset_cfg = cfg.train_datasets
                batch_size = cfg.train_bs
            elif mode == 'val':
                dataset_cfg = cfg.val_datasets
                batch_size = cfg.val_bs
            dataset = MultiLabelDataset(**dataset_cfg)

        elif cfg.dataset_type == 'instance':
            # register all modules in mmdet into the registries
            init_default_scope('mmdet')
            annojson,transform,batch_size = get_mode_cfg(mode, cfg)
            dataset = InstanceDataset(cfg, annojson, transform)

        sampler = DefaultSampler(dataset)
        loader = DataLoader(dataset, 
                pin_memory = True,
                batch_size = batch_size, 
                sampler = sampler,
                collate_fn=custom_collate,
                num_workers=8)
            
        dataloaders.append(loader)

    if len(dataloaders) == 1:
        return dataloaders[0]
    return dataloaders

def get_mode_cfg(mode, cfg):
    if mode == 'train':
        annojson = cfg.train_annojson
        transform = cfg.train_transform
        batch_size = cfg.train_bs
    elif mode == 'val':
        annojson = cfg.val_annojson
        transform = cfg.val_transform
        batch_size = cfg.val_bs
    elif mode == 'test':
        annojson = cfg.test_annojson
        transform = cfg.test_transform
        batch_size = cfg.test_bs
    return annojson,transform,batch_size

def custom_collate(batch):
    images = [item['inputs'] for item in batch]
    data_samples = [item['data_samples'] for item in batch]
    images_tensor = torch.stack(images, dim=0)
    
    return {
        'inputs': images_tensor,
        'data_samples': data_samples,
    }
