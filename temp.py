def make_pretrain_ckpt():
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

    model = FusionNet(args)
    model_ckpt = {}
    smartccs_ckpt = torch.load('checkpoints/CCS_vitl_100M.pth', map_location="cpu", weights_only=True)["teacher"]
    for key,value in smartccs_ckpt.items():
        if 'backbone' in key:
            key = '.'.join(key.split('.')[1:])
            model_ckpt['vit_module.'+key] = value
    load_result = model.load_state_dict(model_ckpt, strict=False)
    print(load_result)
    sam_ckpt = torch.load('checkpoints/sam_vit_h_4b8939.pth', map_location="cpu")
    for key,value in sam_ckpt.items():
        if 'patch_embed' in key:
            key = '.'.join(key.split('.')[1:])
            model_ckpt['dtcwt_module.'+key] = value
    load_result = model.load_state_dict(model_ckpt, strict=False)
    print(load_result)
    torch.save(model.state_dict(), f'checkpoints/fusionnet.pth')
make_pretrain_ckpt()