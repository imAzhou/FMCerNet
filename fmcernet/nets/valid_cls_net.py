import torch.nn as nn
from mmpretrain.models.classifiers import ImageClassifier

num_classes = 2
model_config = dict(
    pretrained = 'checkpoints/resnet50_8xb32_in1k_20210831-ea4938fc.pth',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        frozen_stages=-1,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=num_classes,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ),
    data_preprocessor = dict(
        num_classes=num_classes,
        # RGB format normalization parameters
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        # convert image from BGR to RGB
        to_rgb=True,
    )
)

class ValidClsNet(ImageClassifier):

    def __init__(self):
        ''' '''
        super(ValidClsNet, self).__init__(**model_config)
