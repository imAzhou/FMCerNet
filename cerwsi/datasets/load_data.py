import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from mmengine.dataset.sampler import DefaultSampler
from torch.utils.data.distributed import DistributedSampler
from .cls_dataset import ClsDataset
from .instance_dataset import InstanceDataset

def load_data(cfg, load_modes = []):
    dataloaders = []
    for mode in load_modes:
        if cfg.dataset_type == 'cls':
            loader = load_cls_data(cfg, mode)
        elif cfg.dataset_type == 'instance':
            loader = load_instance_data(cfg, mode)
        dataloaders.append(loader)
    if len(dataloaders) == 1:
        return dataloaders[0]
    else:
        return dataloaders

def load_cls_data(cfg, mode):
    def custom_collate(batch):
        # 拆分 batch 中的图像和标签
        images = [item[0] for item in batch]  # 所有 image_tensor，假设 shape 一致
        image_labels = [item[1] for item in batch]
        metainfo = [item[2] for item in batch]

        # 将 images 转换为一个批次的张量
        images_tensor = torch.stack(images, dim=0)
        imglabels_tensor = torch.as_tensor(image_labels)
        # clsid_mask = torch.stack(clsid_mask, dim=0)
        # multi_pos_label = torch.stack(multi_pos_label, dim=0)

        # 返回一个字典，其中包含张量和不规则的标注信息
        return {
            'images': images_tensor,
            'image_labels': imglabels_tensor,
            'metainfo': metainfo,
        }

    train_transform = transforms.Compose([
        transforms.Resize(cfg.img_size),
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
        transforms.RandomVerticalFlip(p=0.5),    # 随机垂直翻转
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    train_dataset = ClsDataset(cfg.data_root, cfg.train_annojson, train_transform, cfg.classes)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, 
                            pin_memory=True,
                            batch_size=cfg.train_bs, 
                            sampler = train_sampler,
                            # drop_last=True,
                            collate_fn=custom_collate,
                            num_workers=8)
    val_transform = transforms.Compose([
        transforms.Resize(cfg.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    val_dataset = ClsDataset(cfg.data_root, cfg.val_annojson, val_transform, cfg.classes)
    val_sampler = DistributedSampler(val_dataset)
    val_loader = DataLoader(val_dataset, 
                            pin_memory=True,
                            batch_size=cfg.val_bs, 
                            sampler = val_sampler,
                            collate_fn=custom_collate,
                            num_workers=8)
    
    return train_loader, val_loader

def load_instance_data(cfg, mode):
    assert mode in ['train', 'val', 'test']
    def custom_collate(batch):
        images = [item['inputs'] for item in batch]  # 所有 image_tensor
        data_samples = [item['data_samples'] for item in batch]
        image_labels = [item.diagnose for item in data_samples]
        # 将 images 转换为一个批次的张量
        images_tensor = torch.stack(images, dim=0)
        imglabels_tensor = torch.as_tensor(image_labels)
        # 返回一个字典，其中包含张量和不规则的标注信息
        return {
            'inputs': images_tensor,
            'data_samples': data_samples,
            'image_labels': imglabels_tensor,
        }

    if mode == 'train':
        annojson = cfg.train_annojson
        rle_masks_path = cfg.train_rel_file
        transform = cfg.train_transform
        batch_size = cfg.train_bs
    elif mode == 'val':
        annojson = cfg.val_annojson
        rle_masks_path = cfg.val_rel_file
        transform = cfg.val_transform
        batch_size = cfg.val_bs
    elif mode == 'test':
        annojson = cfg.test_annojson
        transform = cfg.test_transform
        batch_size = cfg.test_bs

    dataset = InstanceDataset(cfg, annojson, rle_masks_path, transform)
    sampler = DefaultSampler(dataset)
    loader = DataLoader(dataset, 
                pin_memory=True,
                batch_size=batch_size, 
                sampler = sampler,
                collate_fn=custom_collate,
                num_workers=8)

    return loader
