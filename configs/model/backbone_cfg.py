backbone_cfgdict = {
    'resnet': dict(
        backbone_output_dim = [256, 512, 1024, 2048],
        backbone_ckpt = 'checkpoints/resnet50_8xb32_in1k_20210831-ea4938fc.pth',
        frozen_backbone = False,
        use_peft = None,
        default_input_size = 224
    ),
    'convnext': dict(
        backbone_output_dim = [384, 768, 1536],
        backbone_ckpt = 'checkpoints/convnext-large_in21k-pre-3rdparty_64xb64_in1k_20220124-2412403d.pth',
        frozen_backbone = False,
        use_peft = None,
        default_input_size = 224
    ),
    'ctranspath': dict(
        backbone_output_dim = [768],
        backbone_ckpt = 'checkpoints/ctranspath.pth',
        frozen_backbone = False,
        use_peft = None,
    ),
    'dinov2': dict(
        backbone_output_dim = [1024],
        backbone_ckpt = 'checkpoints/vit-large-p14_dinov2-pre_3rdparty_20230426-f3302d9e.pth',
        frozen_backbone = False,
        use_peft = None,   # None, lora, FourierFT, dtcwt
        vit_patch_size = 14,
        default_input_size = 518
    ),
    'uni': dict(
        backbone_output_dim = [1024],
        backbone_ckpt = 'checkpoints/uni.bin',
        frozen_backbone = True,
        use_peft = 'lora',   # None, lora, FourierFT, dtcwt
        vit_patch_size = 16,
        default_input_size = 224
    ),
    'sam': dict(
        backbone_output_dim = [1024],   # vit_b: 768, vit_l: 1024, vit_h: 1280
        backbone_ckpt = 'checkpoints/sam_vit_l_0b3195.pth',
        backbone_size_type = 'vit_l',
        frozen_backbone = True,
        use_peft = 'lora', 
        use_dtcwt_indexes = [],
        vit_patch_size = 16,
        default_input_size = 1024
    ),
    'sam2': dict(
        backbone_output_dim = [144, 288, 576, 1152],
        config_file = 'cerwsi/nets/backbone/SAM2/configs/sam2.1/sam2.1_hiera_l.yaml',
        backbone_ckpt = 'checkpoints/sam2.1_hiera_large.pt',
        frozen_backbone = True,
        use_peft = 'lora', 
        use_dtcwt_indexes = range(2),  # stage num_blocks: [2,6,36,4]
        default_input_size = 1024
    ),
    'smartccs': dict(
        backbone_output_dim = [1024],
        backbone_ckpt = 'checkpoints/CCS_vitl_100M.pth',
        frozen_backbone = True,
        use_peft = 'lora', 
        default_input_size = 224
    ),
}