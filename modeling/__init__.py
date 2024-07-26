# encoding: utf-8

def build_model(cfg):
    if cfg.MODEL.RENDER =='TEST':
        from .rfrender_blend_v2 import RFRender
        model = RFRender(cfg.MODEL.COARSE_RAY_SAMPLING, cfg.MODEL.FINE_RAY_SAMPLING, boarder_weight= cfg.MODEL.BOARDER_WEIGHT, sample_method = cfg.MODEL.SAMPLE_METHOD, same_space_net = cfg.MODEL.SAME_SPACENET,
                        TriKernel_include_input = cfg.MODEL.TKERNEL_INC_RAW, cfg=cfg)

    elif cfg.MODEL.RENDER =='v1':
        from .rfrender_blend_v1 import RFRender
        model = RFRender(cfg.MODEL.COARSE_RAY_SAMPLING, cfg.MODEL.FINE_RAY_SAMPLING, boarder_weight= cfg.MODEL.BOARDER_WEIGHT, sample_method = cfg.MODEL.SAMPLE_METHOD, same_space_net = cfg.MODEL.SAME_SPACENET,
                        TriKernel_include_input = cfg.MODEL.TKERNEL_INC_RAW, cfg=cfg)

    elif cfg.MODEL.RENDER =='atten':
        from .rfrender_atten import RFRender
        model = RFRender(cfg.MODEL.COARSE_RAY_SAMPLING, cfg.MODEL.FINE_RAY_SAMPLING, boarder_weight= cfg.MODEL.BOARDER_WEIGHT, sample_method = cfg.MODEL.SAMPLE_METHOD, same_space_net = cfg.MODEL.SAME_SPACENET,
                        TriKernel_include_input = cfg.MODEL.TKERNEL_INC_RAW, cfg=cfg)
    else:
        from .rfrender import RFRender
        model = RFRender(cfg.MODEL.COARSE_RAY_SAMPLING, cfg.MODEL.FINE_RAY_SAMPLING, boarder_weight= cfg.MODEL.BOARDER_WEIGHT, sample_method = cfg.MODEL.SAMPLE_METHOD, same_space_net = cfg.MODEL.SAME_SPACENET,
                        TriKernel_include_input = cfg.MODEL.TKERNEL_INC_RAW, cfg=cfg)
    return model


def build_extractor(cfg):
    from .unet import UNet
    extractor = UNet(cfg, n_channels = 4)
    return extractor