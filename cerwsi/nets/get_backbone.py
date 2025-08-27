from .backbone.resnet import ResNet
from .backbone.convnext import ConvNeXt
from .backbone.vit import ViT
from .backbone.dinov2 import DINOV2
from .backbone.uni import UNI
from .backbone.ctranspath import CTransPath
from .backbone.SVT_backbone import SVTBackbone
from .backbone.SAM_backbone import SAMEncoder
from .backbone.SAM2_backbone import SAM2Encoder
from .backbone.SmartCCS_backbone import SmartCCS
from .backbone.FusionNet.fusionnet import FusionNet

allowed_backbone_type = ['resnet', 'convnext', 'vit', 'smartccs', 'fusionnet',
                         'dinov2', 'uni', 'ctranspath', 'svt', 'sam', 'sam2']

def get_backbone(args):
    backbone_type = args.backbone_type
    assert backbone_type in allowed_backbone_type, f'backbone_type allowed in {allowed_backbone_type}'
    
    backbone = None
    if backbone_type == 'resnet':
        backbone = ResNet
    if backbone_type == 'convnext':
        backbone = ConvNeXt
    if backbone_type == 'vit':
        backbone = ViT
    if backbone_type == 'dinov2':
        backbone = DINOV2
    if backbone_type == 'uni':
        backbone = UNI
    if backbone_type == 'ctranspath':
        backbone = CTransPath
    if backbone_type == 'svt':
        backbone = SVTBackbone
    if backbone_type == 'sam':
        backbone = SAMEncoder
    if backbone_type == 'sam2':
        backbone = SAM2Encoder
    if backbone_type == 'smartccs':
        backbone = SmartCCS
    if backbone_type == 'fusionnet':
        backbone = FusionNet
    
    return backbone(args)