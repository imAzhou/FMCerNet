from .backbone.resnet import ResNet
from .backbone.convnext import ConvNeXt
from .backbone.vit import ViT
from .backbone.dinov2 import DINOV2
from .backbone.dinov3 import DINOV3
from .backbone.uni import UNI, UNI2H
from .backbone.virchow import Virchow, Virchow2
from .backbone.gpfm import GPFM
from .backbone.genbio_pathfm import GenBioPathFM
from .backbone.ctranspath import CTransPath
from .backbone.SmartCCS_backbone import SmartCCS
from .backbone.FusionNet.fusionnet import FusionNet
from .backbone.CytoFM_backbone import CytoFM
from .backbone.UniCAS_backbone import UniCAS

allowed_backbone_type = ['resnet', 'convnext', 'vit', 'smartccs', 'fusionnet', 'cytofm', 'unicas', 'dinov2', 'dinov3', 'uni', 'uni2-h', 'virchow', 'virchow2', 'gpfm', 'genbio-pathfm', 'ctranspath']

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
    if backbone_type == 'dinov3':
        backbone = DINOV3
    if backbone_type == 'uni':
        backbone = UNI
    if backbone_type == 'uni2-h':
        backbone = UNI2H
    if backbone_type == 'virchow':
        backbone = Virchow
    if backbone_type == 'virchow2':
        backbone = Virchow2
    if backbone_type == 'gpfm':
        backbone = GPFM
    if backbone_type == 'genbio-pathfm':
        backbone = GenBioPathFM
    if backbone_type == 'ctranspath':
        backbone = CTransPath
    if backbone_type == 'smartccs':
        backbone = SmartCCS
    if backbone_type == 'fusionnet':
        backbone = FusionNet
    if backbone_type == 'cytofm':
        backbone = CytoFM
    if backbone_type == 'unicas':
        backbone = UniCAS
    
    return backbone(args)
