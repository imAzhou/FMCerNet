from torch.utils.data import DataLoader
from cerwsi.datasets import TokenClsDataset
from cerwsi.nets import MultiPatchUNI
import torch
from tqdm import tqdm

# data_root = '/nfs5/zly/codes/CerWSI/data_resource/0103'
data_root = '/x22201018/datasets/CervicalDatasets/data_resource'

def load_data():
    def custom_collate(batch):
        # 拆分 batch 中的图像和标签
        images = [item[0] for item in batch]  # 所有 image_tensor，假设 shape 一致
        image_labels = [item[1] for item in batch]
        token_labels = [item[2] for item in batch]

        # 将 images 转换为一个批次的张量
        images_tensor = torch.stack(images, dim=0)
        imglabels_tensor = torch.as_tensor(image_labels)

        # 返回一个字典，其中包含张量和不规则的标注信息
        return {
            'images': images_tensor,
            'image_labels': imglabels_tensor,
            'token_labels': token_labels  # 保持 label 的原始列表形式
        }

    train_dataset = TokenClsDataset(data_root, 'train')
    train_loader = DataLoader(train_dataset, 
                            batch_size=16, 
                            shuffle=True, 
                            persistent_workers = True,
                            pin_memory=True,
                            collate_fn=custom_collate,
                            num_workers=16)
    val_dataset = TokenClsDataset(data_root, 'val')
    val_loader = DataLoader(val_dataset, 
                            batch_size=16, 
                            shuffle=True, 
                            persistent_workers = True,
                            pin_memory=True,
                            collate_fn=custom_collate,
                            num_workers=16)
    return train_loader, val_loader

def ratio4each_cls():
    abnormal_cnt = 0    # 是阳性图片但没有任何一个阳性 token
    total_cls_cnt = [0]*num_classes   # 记录每个类别的 token 样本数量
    batches_cls_ratio = []  # 每个item代表一个minibatch，记录batch中每个类别的正/负样本比例
    for idx,databatch in enumerate(tqdm(dataloader)):

        bs = len(databatch['image_labels'])
        num_tokens = 14*14
        batch_gt = torch.zeros((num_classes, bs, num_tokens))
        
        for img_label,token_labels,bidx in zip(databatch['image_labels'],databatch['token_labels'],range(bs)):
            if img_label == 1 and len(token_labels) == 0:
                abnormal_cnt += 1
            
            if len(token_labels) > 0:
                for row,col,clsid in token_labels:
                    total_cls_cnt[clsid] += 1
                    # 对于阳性图片来说，标记了类别的token记为1，其他为0的只是没标记，不代表没病变
                    batch_gt[clsid, bidx, (row*14)+col] += 1
            if img_label == 0:
                # 对于阴性图片来说，它所有 token 对于类别0来说都是正样本
                batch_gt[0, bidx, :] += 1
        
        mini_ratio = [] # 当前batch内的所有token各类别正负样本比例
        for i in range(num_classes):
            if i == 0:
                postive_nums = torch.sum(batch_gt[0,...],(0,1))
                negative_nums = torch.sum(batch_gt[1:,...],(0,1,2))
            else:
                postive_nums = torch.sum(batch_gt[i,...],(0,1))
                negative_nums = torch.sum(batch_gt[0,...],(0,1))
            mini_ratio.append((postive_nums/(negative_nums+1e-5)).item())

        batches_cls_ratio.append(mini_ratio)
    
    torch_cls_ratio = torch.tensor(batches_cls_ratio)
    mean_v = torch.mean(torch_cls_ratio, dim=0)
    
    print(f'abnormal cnt: {abnormal_cnt}')
    print(f'cls cnt: {total_cls_cnt}')
    print(f'each class pos/neg token ratio: {mean_v.tolist()}')

def ratio4all_cls():
    mini_batch_cls_ratio = []
    for idx,databatch in enumerate(tqdm(dataloader)):
        bs = len(databatch['image_labels'])
        num_tokens = 14*14
        cls_cnt = [0]*num_classes
        for img_label,token_labels,bidx in zip(databatch['image_labels'],databatch['token_labels'],range(bs)):
            if len(token_labels) > 0:
                for row,col,clsid in token_labels:
                    cls_cnt[clsid] += 1
            if img_label == 0:
                cls_cnt[0] = num_tokens
        mini_batch_cls_ratio.append(
            [x / sum(cls_cnt) for x in cls_cnt]
        )
    
    torch_cls_ratio = torch.tensor(mini_batch_cls_ratio)
    mean_v = torch.mean(torch_cls_ratio, dim=0)
    print(f'each class token ratio: {mean_v.tolist()}')

if __name__ == '__main__':
    train_loader, val_loader = load_data()
    num_classes = 6

    for dataloader in [train_loader, val_loader]:
        # ratio4each_cls()
        ratio4all_cls()

'''
train:
abnormal cnt: 3 (是阳性图片但没有任何一个阳性 token)
cls cnt: [0, 161769, 65319, 123602, 76390, 146854]
each class pos/neg token ratio: 
[39.761741638183594, 0.011854249984025955, 0.00481162266805768, 0.009070327505469322, 0.005602799355983734, 0.010798215866088867]

val:
abnormal cnt: 0
cls cnt: [0, 46568, 19856, 25304, 16170, 42969]
each class pos/neg token ratio: 
[39.92052459716797, 0.013312038965523243, 0.005638749338686466, 0.007235287688672543, 0.004645155277103186, 0.012421541847288609]


token 六分类时，minibatch中平均各类别占比
each class token ratio: [0.7680462598800659, 0.06693318486213684, 0.026784084737300873, 0.050709862262010574, 0.030026821419596672, 0.057499825954437256]

each class token ratio: [0.7588387131690979, 0.07589052617549896, 0.031484205275774, 0.04083908721804619, 0.02566014789044857, 0.06728730350732803]
'''