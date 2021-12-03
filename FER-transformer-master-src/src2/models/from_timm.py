import os
import torch
from torch import nn
import timm
import T2TViT
from T2TViT.models.t2t_vit import t2t_vit_14,t2t_vit_24,t2t_vit_t_14,t2t_vit_t_24
from collections import OrderedDict

def make_config(name,
                folder,
                img_size,
                feature_size,
                load_function=timm.create_model):
    path = os.path.join('./src//models', folder, name)
    return {
        'img_size': img_size,
        'feature_size': feature_size,
        'path': path,
        'load_function': load_function,
    }


def alternate_loading(name, pretrained=True, num_classes=0):
    model = timm.create_model(name, pretrained=pretrained, num_classes=7)
    model.reset_classifier(num_classes=num_classes)
    return model

def t2t_loading(name, pretrained=True, num_classes=0):
    if name == 't2t_14':
        model = t2t_vit_14()
    elif name == 't2t_24':
        model = t2t_vit_24()
    elif name == 't2t_t_14':
        model = t2t_vit_t_14()
    elif name == 't2t_t_24':
        model = t2t_vit_t_24()
    else:
        print('There is no such model here ! OYE (°_°)')
    if pretrained:
        checkpoint = torch.load('/mnt/DONNEES/FER-transformer-master-src/src/models/t2t/'+name+'.pth.tar', map_location='cpu')
        state_dict_key = 'state_dict'
        state_dict_key = 'state_dict_ema'
        new_state_dict = OrderedDict()
        for k, v in checkpoint[state_dict_key].items():
            # strip `module.` prefix
            name = k[7:] if k.startswith('module') else k
            new_state_dict[name] = v
        state_dict = new_state_dict
        model.load_state_dict(state_dict)
    return model


timm_config = {
    # VIT
    'vit_small_patch16_224':
    make_config(
        name='vit_s_16',
        folder='vit',
        img_size=(224, 224),
        feature_size=768,
    ),
    'vit_base_patch16_224':
    make_config(
        name='vit_b_16',
        folder='vit',
        img_size=(224, 224),
        feature_size=768,
    ),
    'vit_large_patch16_224':
    make_config(
        name='vit_l_16',
        folder='vit',
        img_size=(224, 224),
        feature_size=1024,
    ),

    # DEIT
    'vit_deit_small_patch16_224':
    make_config(
        name='deit_s_16',
        folder='deit',
        img_size=(224, 224),
        feature_size=384,
    ),
    'vit_deit_base_patch16_224':
    make_config(
        name='deit_b_16',
        folder='deit',
        img_size=(224, 224),
        feature_size=768,
    ),

    #VIT Resnetv2
    'vit_base_r50_s16_224_in21k':
    make_config(
        name='vit_res_b_16',
        folder='vit-res',
        img_size=(224, 224),
        feature_size=768,
    ),

    # TNT
    'tnt_s_patch16_224':
    make_config(
        name='tnt_s_16',
        folder='tnt',
        img_size=(224, 224),
        feature_size=384,
    ),
    'tnt_b_patch16_224':
    make_config(
        name='tnt_b_16',
        folder='tnt',
        img_size=(224, 224),
        feature_size=640,
    ),

    # CAIT
    'cait_s24_224':
    make_config(
        name='cait_s_24',
        folder='cait',
        img_size=(224, 224),
        feature_size=384,
    ),
    'cait_s36_384':
    make_config(
        name='cait_s_36',
        folder='cait',
        img_size=(384, 384),
        feature_size=384,
    ),
    'cait_m36_384':
    make_config(
        name='cait_m_36',
        folder='cait',
        img_size=(384, 384),
        feature_size=768,
    ),

    # LeViT
    # 'levit_256':
    # make_config(
    #     name='levit_256',
    #     folder='levit',
    #     img_size=(256, 256),
    #     feature_size=384,
    # ),
    # 'levit_384':
    # make_config(
    #     name='levit_384',
    #     folder='levit',
    #     img_size=(384, 384),
    #     feature_size=768,
    # ),

    # ConViT
    #'convit_small':
    #make_config(
    #    name='convit_s',
    #    folder='convit',
    #    img_size=(224, 224),
    #    feature_size=432,
    #),
    #'convit_base':
    #make_config(
    #    name='convit_b',
    #    folder='convit',
    #    img_size=(224, 224),
    #    feature_size=768,
    #),
    
    #T2T
    't2t_14':
    make_config(
        name='t2t_14',
        folder='t2t',
        img_size=(224, 224),
        feature_size=384,
        load_function=t2t_loading,
    ),
    't2t_24':
    make_config(
        name='t2t_24',
        folder='t2t',
        img_size=(224, 224),
        feature_size=512,
        load_function=t2t_loading,
    ),
    't2t_t_14':
    make_config(
        name='t2t_t_14',
        folder='t2t',
        img_size=(224, 224),
        feature_size=384,
        load_function=t2t_loading,
    ),
    't2t_t_24':
    make_config(
        name='t2t_t_24',
        folder='t2t',
        img_size=(224, 224),
        feature_size=512,
        load_function=t2t_loading,
    ),

    # CoaT
    'coat_mini':
    make_config(
        name='coat_mini',
        folder='coat',
        img_size=(224, 224),
        feature_size=216,
        load_function=alternate_loading,
    ),
    'coat_lite_small':
    make_config(
        name='coat_lite_small',
        folder='coat',
        img_size=(224, 224),
        feature_size=512,
        load_function=alternate_loading,
    ),
    
    # Swin-T
    'swin_small_patch4_window7_224':
    make_config(
        name='swin_s_p4w7',
        folder='swin',
        img_size=(224, 224),
        feature_size=768,
    ),
    'swin_base_patch4_window7_224':
    make_config(
        name='swin_b_p4w7',
        folder='swin',
        img_size=(224, 224),
        feature_size=1024,
    ),
    'swin_large_patch4_window7_224':
    make_config(
        name='swin_l_p4w7',
        folder='swin',
        img_size=(224, 224),
        feature_size=1536,
    ),
    
    # CNN
    'inception_resnet_v2':
    make_config(
        name='irv2',
        folder='cnn',
        img_size=(299, 299),
        feature_size=1536,
        load_function=alternate_loading,
    ),
    'xception':
    make_config(
        name='xception',
        folder='cnn',
        img_size=(299, 299),
        feature_size=2048,
    ),
}


class Model(nn.Module):
    def __init__(self, model, config, nb_class=7):

        super(Model, self).__init__()
        self.img_size = config['img_size']
        self.feature_size = config['feature_size']
        self.model = model
        self.model.head = nn.Identity()
        self.nb_class = nb_class
        self.head = nn.Linear(config['feature_size'], nb_class, bias=True)
        #self.activation = activation

    def forward_feat(self, x):
        try:
            return self.model.forward(x)
        except:
            return self.model(x)

    def forward(self, x):
        out_feat = self.forward_feat(x)
        out = self.head(out_feat)
        #out = self.activation(out)
        torch.cuda.device_of(x)
        return out




if __name__ == '__main__':
    for name, config in timm_config.items():
        print("#" * 10, name)
        model = config['load_function'](name, pretrained=True, num_classes=0)
        model = Model(model, config)
        with open(config['path'] + '_config.txt', "a") as f:
            f.write(str(model))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        test_im = torch.randn((
            1,
            3,
            config['img_size'][0],
            config['img_size'][1],
        )).to(device)
        print('image_size:  ', config['img_size'])
        print('hidden_size: ',
              model.forward_feat(test_im).size(1), config['feature_size'])
        print('num_parms:   ', sum(p.numel() for p in model.parameters()))
        torch.save(model, config['path']+'.pth')
        model= torch.load(config['path']+'.pth')
