import torch
from cerwsi.nets import PatchNet
from mmengine.config import Config

device = torch.device(f'cpu')

ckpt = 'checkpoints/detr_r50_8xb2-150e_coco_20221023_153551-436d03e8.pth'
params_weight = torch.load(ckpt, map_location=device, weights_only=False)['state_dict']

new_state_dict = {}
for key,value in params_weight.items():
    if 'bbox_head.fc_cls' in key:
        continue
    if 'backbone' in key:
        new_name = key.replace('backbone.', 'backbone.backbone.')
    else:
        new_name = 'taskhead.' + key
    new_state_dict[new_name] = value

torch.save(new_state_dict, f'checkpoints/detr_r50_rename.pth')

# d_cfg = Config.fromfile('configs/dataset/mmdet/hmchh_dataset.py')
# m_cfg = Config.fromfile('configs/model/detr.py')
# s_cfg = Config.fromfile('configs/strategy.py')

# cfg = Config()
# for sub_cfg in [d_cfg, m_cfg, s_cfg]:
#     cfg.merge_from_dict(sub_cfg.to_dict())
# cfg.save_result_dir = None
# model = PatchNet(cfg).to(device)
# print(model.load_state_dict(new_state_dict, strict=False))
