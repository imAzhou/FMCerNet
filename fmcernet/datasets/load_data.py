import torch
import math
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
from mmengine.registry import init_default_scope
from mmengine.dataset.sampler import DefaultSampler
from .slide_dataset import SlideDataset
from mmpretrain.datasets import MultiLabelDataset,CustomDataset


class BalancedBatchSampler(Sampler):
    """Yield indices ordered so each DataLoader batch is roughly pos:neg balanced."""

    def __init__(self, labels, batch_size, pos_fraction=0.5, seed=1234):
        self.labels = torch.as_tensor(labels).long()
        self.batch_size = batch_size
        self.pos_per_batch = int(round(batch_size * pos_fraction))
        self.pos_per_batch = min(max(self.pos_per_batch, 1), batch_size - 1)
        self.neg_per_batch = batch_size - self.pos_per_batch
        self.seed = seed
        self.epoch = 0

        self.pos_indices = torch.where(self.labels == 1)[0]
        self.neg_indices = torch.where(self.labels == 0)[0]
        if len(self.pos_indices) == 0 or len(self.neg_indices) == 0:
            raise ValueError(
                "BalancedBatchSampler requires both positive and negative samples."
            )

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self.rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

        raw_global_batches = math.ceil(len(self.labels) / self.batch_size)
        self.num_batches = math.ceil(raw_global_batches / self.world_size)
        self.global_num_batches = self.num_batches * self.world_size
        self.num_samples = self.num_batches * self.batch_size

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _draw_indices(self, indices, num_samples, generator):
        chunks = []
        remaining = num_samples
        while remaining > 0:
            perm = torch.randperm(len(indices), generator=generator)
            take = min(remaining, len(indices))
            chunks.append(indices[perm[:take]])
            remaining -= take
        return torch.cat(chunks)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch * 10007)

        pos_needed = self.global_num_batches * self.pos_per_batch
        neg_needed = self.global_num_batches * self.neg_per_batch
        pos_order = self._draw_indices(self.pos_indices, pos_needed, generator)
        neg_order = self._draw_indices(self.neg_indices, neg_needed, generator)

        all_indices = []
        for idx in range(self.global_num_batches):
            if idx % self.world_size != self.rank:
                continue
            pos = pos_order[idx * self.pos_per_batch:(idx + 1) * self.pos_per_batch]
            neg = neg_order[idx * self.neg_per_batch:(idx + 1) * self.neg_per_batch]
            batch = torch.cat([pos, neg])
            batch = batch[torch.randperm(len(batch), generator=generator)]
            all_indices.extend(batch.tolist())
        return iter(all_indices)

    def __len__(self):
        return self.num_samples


def get_binary_labels(dataset):
    labels = []
    for idx in range(len(dataset)):
        data_info = dataset.get_data_info(idx)
        gt_label = data_info.get('gt_label', [])
        labels.append(int(len(gt_label) > 0))
    return labels


def build_sampler(cfg, mode, dataset, batch_size):
    sampler_cfg = cfg.get('train_sampler', {}) if mode == 'train' else {}
    if sampler_cfg.get('type', None) == 'balanced_batch':
        labels = get_binary_labels(dataset)
        return BalancedBatchSampler(
            labels,
            batch_size=batch_size,
            pos_fraction=sampler_cfg.get('pos_fraction', 0.5),
            seed=sampler_cfg.get('seed', 1234),
        )
    return DefaultSampler(dataset)


def load_data(cfg, load_modes = []):
    valid_modes = {'train', 'val', 'test'}

    assert all(mode in valid_modes for mode in load_modes), \
    f"Invalid mode(s) in load_modes: {load_modes}. Must be in {valid_modes}"

    dataloaders = []
    for mode in load_modes:
        if cfg.dataset_type == 'cls':
            init_default_scope('mmpretrain')
            dataset_cfg = {}
            if mode == 'train':
                dataset_cfg = cfg.train_datasets
                batch_size = cfg.train_bs
            elif mode == 'val':
                dataset_cfg = cfg.val_datasets
                batch_size = cfg.val_bs
            dataset = CustomDataset(**dataset_cfg)
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
        
        elif cfg.dataset_type == 'slide':
            if mode == 'train':
                csvfile = cfg.train_csvfile
                batch_size = cfg.train_bs
            elif mode == 'val':
                csvfile = cfg.val_csvfile
                batch_size = cfg.val_bs
            dataset = SlideDataset(cfg, csvfile)

        sampler = build_sampler(cfg, mode, dataset, batch_size)
        loader = DataLoader(dataset, 
                pin_memory = True,
                batch_size = batch_size, 
                sampler = sampler,
                collate_fn = custom_collate,
                num_workers = 8)
            
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
