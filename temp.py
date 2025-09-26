import torch
from thop import profile
from thop.vision.basic_hooks import zero_ops
from cerwsi.nets import PatchNet
from mmengine.config import Config
import peft
from mmpretrain.structures import DataSample

device = torch.device('cuda:0')
config_file = 'log/WS1200/query2label/2025_09_25_09_37_47/config.py'
cfg = Config.fromfile(config_file)
model = PatchNet(cfg).to(device)
input = torch.randn(1, 3, 448, 448).to(device)  # batch=1, 3通道, 224x224输入


data_batch = {
    'inputs': input,
    'data_samples':[DataSample()]
}
flops, params = profile(model, inputs=(data_batch, 'val'))
print(f"FLOPs: {flops/1e9:.2f} GFLOPs")
print(f"Params: {params/1e6:.2f} M")

'''
Ours: 
FLOPs: 90.21 GFLOPs
Params: 326.76 M

ml_decoder 224 / 448:
FLOPs: 78.85 / 314.42 GFLOPs
Params: 310.01 M

query2label 224 / 448:
FLOPs: 88.10 / 350.46 GFLOPs
Params: 408.90 M
'''