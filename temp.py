from cerwsi.nets.backbone.SVT_backbone import SVTBackbone
import torch

device = torch.device('cuda:1')
model = SVTBackbone().to(device)
inputs = torch.randn(7,3,224,224).to(device)
output = model(inputs)
print()