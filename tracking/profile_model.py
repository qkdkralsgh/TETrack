import argparse
import torch
import os
import time
import importlib
from thop import profile
from thop.utils import clever_format

def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Parse args for training')
    # for train
    parser.add_argument('--script', type=str, default='mixformer_online', choices=['mixformer_vit', 'mixformer_online', 'mixformer_cvt'],
                        help='training script name')
    parser.add_argument('--config', type=str, default='baseline', help='yaml configure file name')
    parser.add_argument('--display_name', type=str, default='MixFormer')
    parser.add_argument('--online_skip', type=int, default=200, help='the skip interval of mixformer-online')
    args = parser.parse_args()

    return args


def evaluate(model, template, search, display_info='MixFormer'):
    """Compute FLOPs, Params, and Speed"""
    # compute flops and params except for score prediction
    macs, params = profile(model, inputs=(template, template, search),  ### Edit by Minho 2023.07.29
                           custom_ops=None, verbose=False)
    macs, params = clever_format([macs, params], "%.3f")
    print('==>Macs is ', macs)
    print('==>Params is ', params)

    # test speed
    T_w = 500
    T_t = 1000
    print("testing speed ...")
    with torch.no_grad():
        # overall
        for i in range(T_w):
            _ = model(template, template, search)
        start = time.time()
        for i in range(T_t):
            _ = model(template, template, search)
        end = time.time()
        avg_lat = (end - start) / T_t
        print("\033[0;32;40m The average overall FPS of {} is {}.\033[0m" .format(display_info, 1.0/avg_lat))


def get_data(bs, sz):
    img_patch = torch.randn(bs, 3, sz, sz)
    return img_patch


if __name__ == "__main__":
    device = "cuda:0"
    torch.cuda.set_device(device)
    args = parse_args()
    '''update cfg'''
    prj_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    yaml_fname = os.path.join(prj_dir, 'experiments/%s/%s.yaml' % (args.script, args.config))
    print("yaml_fname: {}".format(yaml_fname))
    config_module = importlib.import_module('lib.config.%s.config' % args.script)
    cfg = config_module.cfg
    config_module.update_config_from_file(yaml_fname)
    print("cfg: {}".format(cfg))
    '''set some values'''
    bs = 1
    z_sz = cfg.TEST.TEMPLATE_SIZE
    x_sz = cfg.TEST.SEARCH_SIZE
    cfg.MODEL.BACKBONE.FREEZE_BN = False
    cfg.MODEL.HEAD_FREEZE_BN = False
    '''import stark network module'''
    model_module = importlib.import_module('lib.models.%s.tetrack_vit' % args.script)
    model_constructor = model_module.build_tetrack_vit
    model = model_constructor(cfg)
    # get the template and search
    template = get_data(bs, z_sz)
    search = get_data(bs, x_sz)
    # transfer to device
    model = model.to(device)
    template = template.to(device)
    search = search.to(device)
    # evaluate the model properties
    evaluate(model, template, search, args.display_name)
