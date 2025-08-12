from cerwsi.nets import PatchNet,ValidClsNet
from mmengine.config import Config
import torch
from mmpretrain.structures import DataSample

pnmodel_rootdir = 'log/WS850/hs_round0'
mmcls_config_file = f'{pnmodel_rootdir}/config.py'
mmcls_ckpt = f'{pnmodel_rootdir}/checkpoints/best.pth'
test_bs = 64

device = torch.device('cuda:1')

cfg = Config.fromfile(mmcls_config_file)
cfg.backbone_cfg['backbone_ckpt'] = None
mlcls_model = PatchNet(cfg).to(device)
mlcls_model.img_size = cfg.input_size
mlcls_model.load_ckpt(mmcls_ckpt)
mlcls_model.eval()

for i in range(10):
    inputs = torch.rand(test_bs, 3, mlcls_model.img_size, mlcls_model.img_size).to(device)
    data_batch = dict(inputs=inputs, data_samples=[DataSample() for _ in range(test_bs)])
    with torch.no_grad():
        outputs = mlcls_model(data_batch, 'val')
print()
