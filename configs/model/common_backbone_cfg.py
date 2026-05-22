common_backbone_cfgdict = {
    'resnet': dict(
        backbone_output_dim = [256, 512, 1024, 2048],
        backbone_token_output_dim = [2048],
        backbone_ckpt = 'checkpoints/resnet50_8xb32_in1k_20210831-ea4938fc.pth',
        frozen_backbone = False,
        use_peft = None,
        default_input_size = 224
    ),
    'convnext': dict(
        backbone_output_dim = [256, 512, 1024],
        backbone_token_output_dim = [1024],
        backbone_output_downratio = [8, 16, 32],
        backbone_ckpt = 'checkpoints/dinov3-convnext-base-pretrain-lvd1689m',
        out_indices = (2, 3, 4),
        frozen_backbone = False,
        use_peft = None,
        default_input_size = 224
    ),
    'vit': dict(
        backbone_output_dim = [1024],
        backbone_token_output_dim = [1024],
        backbone_output_downratio = [16],
        backbone_ckpt = 'checkpoints/vit-large-p16_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-b20ba619.pth',
        frozen_backbone = False,
        use_peft = None,   # None, lora, FourierFT, dtcwt
        lora_target_modules = ["qkv", "proj", "fc1", "fc2"],
        vit_patch_size = 16,
        default_input_size = 224
    ),
    'dinov2': dict(
        backbone_output_dim = [1024],
        backbone_token_output_dim = [1024],
        backbone_output_downratio = [14],
        backbone_ckpt = 'checkpoints/vit-large-p14_dinov2-pre_3rdparty_20230426-f3302d9e.pth',
        frozen_backbone = False,
        use_peft = None,   # None, lora, FourierFT, dtcwt
        lora_target_modules = ["qkv", "proj", "fc1", "fc2"],
        vit_patch_size = 14,
        default_input_size = 518
    ),
    'dinov3': dict(
        backbone_output_dim = [1024],
        backbone_token_output_dim = [1024],
        backbone_output_downratio = [16],
        backbone_ckpt = 'checkpoints/dinov3-vitl16-pretrain-lvd1689m',
        frozen_backbone = False,
        use_peft = None,
        lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
        vit_patch_size = 16,
        default_input_size = 224
    ),
    
}
