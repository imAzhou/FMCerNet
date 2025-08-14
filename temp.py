import torch
from cerwsi.nets.backbone.FusionNet.fusionnet import FusionNet
import argparse
parser = argparse.ArgumentParser()
args = parser.parse_args()
args.input_size = 1024
args.backbone_cfg = {
    'use_peft': None,
    'frozen_backbone': True,
    'backbone_ckpt': None
}

device = torch.device('cuda:0')
model = FusionNet(args).to(device)
data = torch.randn((7, 3, 1024, 1024)).to(device)
output = model(data)

print()