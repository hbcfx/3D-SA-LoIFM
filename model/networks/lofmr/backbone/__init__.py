from .resnet_fpn import ResNetFPN_8,ResNetFPN_8_2, ResNetFPN_16_4,ResNetFPN_4,ResNetFPN_2,ResNetFPN

def build_backbone(config):
    if config['BACKBONE_TYPE'] == 'ResNetFPN':
        if config['RESOLUTION'] == (1):
            return ResNetFPN(config['RESNETFPN'])
        elif config['RESOLUTION'] == (2):
            return ResNetFPN_2(config['RESNETFPN'])
        elif config['RESOLUTION'] == (4):
            return ResNetFPN_4(config['RESNETFPN'])
        elif config['RESOLUTION'] == (8):
            return ResNetFPN_8(config['RESNETFPN'])
        elif config['RESOLUTION'] == (8, 2):
            return ResNetFPN_8_2(config['RESNETFPN'])
        elif config['RESOLUTION'] == (16, 4):
            return ResNetFPN_16_4(config['RESNETFPN'])
    else:
        raise ValueError(f"LOFTR.BACKBONE_TYPE {config['BACKBONE_TYPE']} not supported.")
