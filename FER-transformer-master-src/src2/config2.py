import torch
import os
import models.cvt.src.cvt as cvt
import models.t2t.src.t2t as t2t
from models.T2TViT.models.t2t_vit import t2t_vit_14,t2t_vit_24,t2t_vit_t_14,t2t_vit_t_24
from models import T2TViT
from models.from_timm2 import *

models = {
    'vit_s_16':
    lambda: torch.load(os.path.join('./src2/models', 'vit/vit_s_16.pth'), map_location=torch.device('cpu')),
    'vit_b_16':
    lambda: torch.load(os.path.join('./src2/models', 'vit/vit_b_16.pth'), map_location=torch.device('cpu')),
    'vit_l_16':
    lambda: torch.load(os.path.join('./src2/models', 'vit/vit_l_16.pth'), map_location=torch.device('cpu')),
    'deit_s_16':
    lambda: torch.load(os.path.join('./src2/models', 'deit/deit_s_16.pth'), map_location=torch.device('cpu')),
    'deit_b_16':
    lambda: torch.load(os.path.join('./src2/models', 'deit/deit_b_16.pth'), map_location=torch.device('cpu')),
    'vit_res_b_16':
    lambda: torch.load(os.path.join('./src2/models', 'vit-res/vit_res_b_16.pth'), map_location=torch.device('cpu')),
    'tnt_s_16':
    lambda: torch.load(os.path.join('./src2/models', 'tnt/tnt_s_16.pth'), map_location=torch.device('cpu')),
    'tnt_b_16':
    lambda: torch.load(os.path.join('./src2/models', 'tnt/tnt_b_16.pth'), map_location=torch.device('cpu')),
    'cait_s_24':
    lambda: torch.load(os.path.join('./src2/models', 'cait/cait_s_24.pth'), map_location=torch.device('cpu')),
    'cait_s_36':
    lambda: torch.load(os.path.join('./src2/models', 'cait/cait_s_36.pth'), map_location=torch.device('cpu')),
    'cait_m_36':
    lambda: torch.load(os.path.join('./src2/models', 'cait/cait_m_36.pth'), map_location=torch.device('cpu')),
    'convit_s':
    lambda: torch.load(os.path.join('./src2/models', 'convit/convit_s.pth'), map_location=torch.device('cpu')),
    'convit_b':
    lambda: torch.load(os.path.join('./src2/models', 'convit/convit_b.pth'), map_location=torch.device('cpu')),
    'coat_mini':
    lambda: torch.load(os.path.join('./src2/models', 'coat/coat_mini.pth'), map_location=torch.device('cpu')),
    'coat_lite_small':
    lambda: torch.load(os.path.join('./src2/models', 'coat/coat_lite_small.pth'), map_location=torch.device('cpu')),
    'swin_s_p4w7':
    lambda: torch.load(os.path.join('./src2/models', 'swin/swin_s_p4w7.pth'), map_location=torch.device('cpu')),
    'swin_b_p4w7':
    lambda: torch.load(os.path.join('./src2/models', 'swin/swin_b_p4w7.pth'), map_location=torch.device('cpu')),
    'swin_l_p4w7':
    lambda: torch.load(os.path.join('./src2/models', 'swin/swin_l_p4w7.pth'), map_location=torch.device('cpu')),
    'cvt_21_384':
    lambda: cvt.load_model('cvt_21_384', 'cvt/cvt_21_384.pth'),
    'cvt_w24_384':
    lambda: cvt.load_model('cvt_w24_384', 'cvt/cvt_w24_384.pth'),
    't2t_14':
    lambda: torch.load(os.path.join('./src2/models', 't2t/t2t_14.pth'), map_location=torch.device('cpu')),
    't2t_24':
    lambda: torch.load(os.path.join('./src2/models', 't2t/t2t_24.pth'), map_location=torch.device('cpu')),
    't2t_t_14':
    lambda: torch.load(os.path.join('./src2/models', 't2t/t2t_14.pth'), map_location=torch.device('cpu')),
    't2t_t_24':
    lambda: torch.load(os.path.join('./src2/models', 't2t/t2t_24.pth'), map_location=torch.device('cpu')),
    'irv2':
    lambda: torch.load(os.path.join('./src2/models', 'cnn/irv2.pth'), map_location=torch.device('cpu')),
    'xception':
    lambda: torch.load(os.path.join('./src2/models', 'cnn/xception.pth'), map_location=torch.device('cpu')),
}


def load_model2(name: str) -> torch.nn.Module:
    model = models[name]
    return model()
