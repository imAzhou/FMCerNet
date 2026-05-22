histopathology_backbone_cfgdict = {
    'ctranspath': dict(
        backbone_output_dim = [768],
        backbone_token_output_dim = [768],
        backbone_output_downratio = [32],
        backbone_ckpt = 'checkpoints/ctranspath.pth',
        frozen_backbone = False,
        use_peft = None,
    ),
    'uni': dict(
        backbone_output_dim = [1024],
        backbone_token_output_dim = [1024],
        backbone_output_downratio = [16],
        backbone_ckpt = 'checkpoints/uni.bin',
        frozen_backbone = True,
        use_peft = 'lora',   # None, lora, FourierFT, dtcwt
        lora_target_modules = ["qkv", "proj", "fc1", "fc2"],
        vit_patch_size = 16,
        default_input_size = 224
    ),
    'uni2-h': dict(
        backbone_output_dim = [1536],
        backbone_token_output_dim = [1536],
        backbone_output_downratio = [14],
        backbone_ckpt = 'checkpoints/uni2-h.bin',
        frozen_backbone = True,
        use_peft = 'lora',   # None, lora, FourierFT, dtcwt
        lora_target_modules = ["qkv", "proj", "fc1", "fc2"],
        vit_patch_size = 14,
        num_register_tokens = 8,
        default_input_size = 224
    ),
    'virchow': dict(
        backbone_output_dim = [2560],
        backbone_token_output_dim = [1280],
        backbone_output_downratio = [14],
        backbone_ckpt = 'checkpoints/virchow.safetensors',
        frozen_backbone = True,
        use_peft = None,
        vit_patch_size = 14,
        default_input_size = 224
    ),
    'virchow2': dict(
        backbone_output_dim = [2560],
        backbone_token_output_dim = [1280],
        backbone_output_downratio = [14],
        backbone_ckpt = 'checkpoints/virchow2.safetensors',
        frozen_backbone = True,
        use_peft = None,
        vit_patch_size = 14,
        num_register_tokens = 4,
        default_input_size = 224
    ),
    'gpfm': dict(
        backbone_output_dim = [1024],
        backbone_token_output_dim = [1024],
        backbone_output_downratio = [14],
        backbone_ckpt = 'checkpoints/GPFM.pth',
        checkpoint_key = 'teacher',
        checkpoint_prefix = 'backbone.',
        frozen_backbone = True,
        use_peft = None,
        lora_target_modules = ["qkv", "proj", "fc1", "fc2"],
        vit_patch_size = 14,
        default_input_size = 224
    ),
    'genbio-pathfm': dict(
        backbone_output_dim = [4608],
        backbone_token_output_dim = [4608],
        backbone_output_downratio = [16],
        backbone_ckpt = 'checkpoints/genbio-pathfm.pth',
        frozen_backbone = True,
        use_peft = None,
        lora_target_modules = ["qkv", "proj", "w1", "w2", "w3"],
        vit_patch_size = 16,
        num_register_tokens = 4,
        default_input_size = 224
    ),
}
