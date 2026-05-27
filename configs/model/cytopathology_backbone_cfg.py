cytopathology_backbone_cfgdict = {
    'smartccs': dict(
        backbone_output_dim = [1024],
        backbone_token_output_dim = [1024],
        backbone_output_downratio = [14],
        backbone_ckpt = 'checkpoints/CCS_vitl_100M.pth',
        frozen_backbone = True,
        use_peft = 'lora', 
        lora_target_modules = ["qkv", "proj", "fc1", "fc2"],
        vit_patch_size = 14,
        default_input_size = 224
    ),
    'fusionnet': dict(
        backbone_output_dim = [1024],
        backbone_token_output_dim = [1024],
        backbone_output_downratio = [64],
        backbone_ckpt = 'checkpoints/CCS_vitl_100M.pth',
        frozen_backbone = True,
        use_peft = 'lora', 
        lora_target_modules = ["qkv", "proj", "fc1", "fc2"],
        DTBlock_nums = 3
    ),
    'cytofm': dict(
        backbone_output_dim = [768],
        backbone_token_output_dim = [768],
        backbone_output_downratio = [16],
        backbone_ckpt = 'checkpoints/cytofm_weights.pth',
        frozen_backbone = True,
        use_peft = 'lora',
        lora_target_modules = ["qkv", "proj", "fc1", "fc2"],
        vit_patch_size = 16,
        default_input_size = 224
    ),
    'unicas': dict(
        backbone_output_dim = [1024],
        backbone_token_output_dim = [1024],
        backbone_output_downratio = [16],
        backbone_ckpt = 'checkpoints/UniCAS.pth',
        frozen_backbone = True,
        use_peft = 'lora',
        lora_target_modules = ["qkv", "proj", "w12", "w3"],
        vit_patch_size = 16,
        default_input_size = 224
    ),
}
