import torch
import os
import models.cvt.src.cvt as cvt
import models.t2t.src.t2t as t2t
from models.T2TViT.models.t2t_vit import t2t_vit_14,t2t_vit_24,t2t_vit_t_14,t2t_vit_t_24
from models import T2TViT

models = {
    'vit_s_16':
    lambda: torch.load(os.path.join('./src/models', 'vit/vit_s_16.pth')),
    'vit_b_16':
    lambda: torch.load(os.path.join('./src/models', 'vit/vit_b_16.pth')),
    'vit_l_16':
    lambda: torch.load(os.path.join('./src/models', 'vit/vit_l_16.pth')),
    'deit_s_16':
    lambda: torch.load(os.path.join('./src/models', 'deit/deit_s_16.pth')),
    'deit_b_16':
    lambda: torch.load(os.path.join('./src/models', 'deit/deit_b_16.pth')),
    'vit_res_b_16':
    lambda: torch.load(os.path.join('./src/models', 'vit-res/vit_res_b_16.pth')),
    'tnt_s_16':
    lambda: torch.load(os.path.join('./src/models', 'tnt/tnt_s_16.pth')),
    'tnt_b_16':
    lambda: torch.load(os.path.join('./src/models', 'tnt/tnt_b_16.pth')),
    'cait_s_24':
    lambda: torch.load(os.path.join('./src/models', 'cait/cait_s_24.pth')),
    'cait_s_36':
    lambda: torch.load(os.path.join('./src/models', 'cait/cait_s_36.pth')),
    'cait_m_36':
    lambda: torch.load(os.path.join('./src/models', 'cait/cait_m_36.pth')),
    'convit_s':
    lambda: torch.load(os.path.join('./src/models', 'convit/convit_s.pth')),
    'convit_b':
    lambda: torch.load(os.path.join('./src/models', 'convit/convit_b.pth')),
    'coat_mini':
    lambda: torch.load(os.path.join('./src/models', 'coat/coat_mini.pth')),
    'coat_lite_small':
    lambda: torch.load(os.path.join('./src/models', 'coat/coat_lite_small.pth')),
    'swin_s_p4w7':
    lambda: torch.load(os.path.join('./src/models', 'swin/swin_s_p4w7.pth')),
    'swin_b_p4w7':
    lambda: torch.load(os.path.join('./src/models', 'swin/swin_b_p4w7.pth')),
    'swin_l_p4w7':
    lambda: torch.load(os.path.join('./src/models', 'swin/swin_l_p4w7.pth')),
    'cvt_21_384':
    lambda: cvt.load_model('cvt_21_384', 'cvt/cvt_21_384.pth'),
    'cvt_w24_384':
    lambda: cvt.load_model('cvt_w24_384', 'cvt/cvt_w24_384.pth'),
    't2t_14':
    lambda: torch.load(os.path.join('./src/models', 't2t/t2t_14.pth')),
    't2t_24':
    lambda: torch.load(os.path.join('./src/models', 't2t/t2t_24.pth')),
    't2t_t_14':
    lambda: torch.load(os.path.join('./src/models', 't2t/t2t_14.pth')),
    't2t_t_24':
    lambda: torch.load(os.path.join('./src/models', 't2t/t2t_24.pth')),
    'irv2':
    lambda: torch.load(os.path.join('./src/models', 'cnn/irv2.pth')),
    'xception':
    lambda: torch.load(os.path.join('./src/models', 'cnn/xception.pth')),
}


def load_model(name: str) -> torch.nn.Module:
    model = models[name]
    return model()
