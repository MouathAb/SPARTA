import os
import torch
from torch import nn
import timm
import T2TViT
from T2TViT.models.t2t_vit import t2t_vit_14,t2t_vit_24,t2t_vit_t_14,t2t_vit_t_24
from collections import OrderedDict
from torchvision import transforms
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_
from timm.models.layers.helpers import to_2tuple
from functools import partial
import argparse

def make_config(name,
                folder,
                img_size,
                temp_size,
                feature_size,
                load_function=timm.create_model):
    path = os.path.join('./src2//models', folder, name)
    return {
        'img_size': img_size,
        'temp_size': temp_size,
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
        checkpoint = torch.load('./src2/models/t2t/'+name+'.pth.tar', map_location='cpu')
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
        temp_size = 10,
        feature_size=768,
    ),
    'vit_base_patch16_224':
    make_config(
        name='vit_b_16',
        folder='vit',
        img_size=(224, 224),
        temp_size = 10,
        feature_size=768,
    ),
    'vit_large_patch16_224':
    make_config(
        name='vit_l_16',
        folder='vit',
        img_size=(224, 224),
        temp_size = 10,
        feature_size=1024,
    ),

    # DEIT
    'vit_deit_small_patch16_224':
    make_config(
        name='deit_s_16',
        folder='deit',
        img_size=(224, 224),
        temp_size = 10,
        feature_size=384,
    ),
    'vit_deit_base_patch16_224':
    make_config(
        name='deit_b_16',
        folder='deit',
        img_size=(224, 224),
        temp_size = 10,
        feature_size=768,
    ),

    #VIT Resnetv2
    'vit_base_r50_s16_224_in21k':
    make_config(
        name='vit_res_b_16',
        folder='vit-res',
        img_size=(224, 224),
        temp_size = 10,
        feature_size=768,
    ),

    # TNT
    'tnt_s_patch16_224':
    make_config(
        name='tnt_s_16',
        folder='tnt',
        img_size=(224, 224),
        temp_size = 10,
        feature_size=384,
    ),
    'tnt_b_patch16_224':
    make_config(
        name='tnt_b_16',
        folder='tnt',
        img_size=(224, 224),
        temp_size = 10,
        feature_size=640,
    ),

    # CAIT
    'cait_s24_224':
    make_config(
        name='cait_s_24',
        folder='cait',
        img_size=(224, 224),
        temp_size = 10,
        feature_size=384,
    ),
    'cait_s36_384':
    make_config(
        name='cait_s_36',
        folder='cait',
        img_size=(384, 384),
        temp_size = 10,
        feature_size=384,
    ),
    'cait_m36_384':
    make_config(
        name='cait_m_36',
        folder='cait',
        img_size=(384, 384),
        temp_size = 10,
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
        temp_size = 10,
        feature_size=384,
        load_function=t2t_loading,
    ),
    't2t_24':
    make_config(
        name='t2t_24',
        folder='t2t',
        img_size=(224, 224),
        temp_size = 10,
        feature_size=512,
        load_function=t2t_loading,
    ),
    't2t_t_14':
    make_config(
        name='t2t_t_14',
        folder='t2t',
        img_size=(224, 224),
        temp_size = 10,
        feature_size=384,
        load_function=t2t_loading,
    ),
    't2t_t_24':
    make_config(
        name='t2t_t_24',
        folder='t2t',
        img_size=(224, 224),
        temp_size = 10,
        feature_size=512,
        load_function=t2t_loading,
    ),

    # CoaT
    'coat_mini':
    make_config(
        name='coat_mini',
        folder='coat',
        img_size=(224, 224),
        temp_size = 10,
        feature_size=216,
        load_function=alternate_loading,
    ),
    'coat_lite_small':
    make_config(
        name='coat_lite_small',
        folder='coat',
        img_size=(224, 224),
        temp_size = 10,
        feature_size=512,
        load_function=alternate_loading,
    ),
    
    # Swin-T
    'swin_small_patch4_window7_224':
    make_config(
        name='swin_s_p4w7',
        folder='swin',
        img_size=(224, 224),
        temp_size = 10,
        feature_size=768,
    ),
    'swin_base_patch4_window7_224':
    make_config(
        name='swin_b_p4w7',
        folder='swin',
        img_size=(224, 224),
        temp_size = 10,
        feature_size=1024,
    ),
    'swin_large_patch4_window7_224':
    make_config(
        name='swin_l_p4w7',
        folder='swin',
        img_size=(224, 224),
        temp_size = 10,
        feature_size=1536,
    ),
    
    # CNN
    'inception_resnet_v2':
    make_config(
        name='irv2',
        folder='cnn',
        img_size=(299, 299),
        temp_size = 10,
        feature_size=1536,
        load_function=alternate_loading,
    ),
    'xception':
    make_config(
        name='xception',
        folder='cnn',
        img_size=(299, 299),
        temp_size = 10,
        feature_size=2048,
    ),
}

    
class PatchEmbed3d(nn.Module):
    """
    (3D=2D+time) sequnce of images to Patch Embedding
    """
    def __init__(self,model,img_size=224,temp_embed_size=6, patch_size=16, ptemp_size=6,
                 in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        
        self.img_size = img_size
        self.temp_embed_size = temp_embed_size
        self.ptemp_size = ptemp_size
        self.patch_size = patch_size
        self.grid_size = (temp_embed_size//ptemp_size,img_size[0]//patch_size[0], img_size[1]//patch_size[1])
        self.num_patches = self.grid_size[0]*self.grid_size[1]*self.grid_size[2]
        self.flatten = flatten
        
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=(ptemp_size,patch_size[0],patch_size[1]), stride=(ptemp_size,patch_size[0],patch_size[1]))
        self.norm = model.patch_embed.norm
        
        #initilize the 3D-CNN felters with centrale init "ViViT paper:"
        pre_w = model.patch_embed.proj.weight.data
        o,i,h,w = pre_w.shape
        _w = torch.zeros(ptemp_size,o,i,h,w)
        _w[2,:,:,:,:] = pre_w
        self.proj.weight.data = _w.permute(1,2,0,3,4)
        self.proj.bias.data = model.patch_embed.proj.bias.data
    
    def forward(self, x):
        T,B,C,H,W = x.shape
        assert T == self.temp_embed_size and H == self.img_size[0] and H == self.img_size[1], \
            f"Input Sequence size ({T}*{H}*{W}) doesn't match model ({self.temp_size}*{self.img_size[0]}*{self.img_size[1]})."
        x = x.permute(1,2,0,3,4)
        x = self.proj(x)
        if self.flatten:
            x = torch.squeeze(x,dim=2).flatten(2).transpose(1,2) #BCTHW -> BNC
        
        x = self.norm(x)
        return x
    
class SpatialEncoder(nn.Module):
    def __init__(self, model, config, patch_size=16,  window=10): 

        super(SpatialEncoder, self).__init__()
        self.img_size = config['img_size']
        self.temp_size = config['temp_size']
        self.window = window
        self.temp_embed_size = self.window//self.temp_size # window: is the length of the sliding window through the entire sequence
        self.feature_size = config['feature_size']
        self.embed_dim = self.feature_size
        self.model = model
        
        self.model.head = nn.Identity()
        if window != self.temp_size:
            self.patch_embed = PatchEmbed3d(self.model,img_size=self.img_size,
                                            temp_embed_size=self.temp_embed_size, patch_size=patch_size,
                                            ptemp_size=self.temp_embed_size, in_chans=3, embed_dim=self.embed_dim)        

    def forward_feat(self, x):
        if self.window == self.temp_size:
            x =  self.model.patch_embed(x)
        else: 
            x = self.patch_embed(x)
        cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token,x),dim=1)
        x = self.model.pos_drop(x + self.model.pos_embed)
        x = self.model.blocks(x)
        x = self.model.norm(x)
        '''
        try:     
            out_feat = self.model.forward(x)
            b,c = out_feat.shape
            return out_feat - self.model.cls_token.view(1,c)
        except:
            out_feat = self.model(x)
            b,c = out_feat.shape
            return out_feat - self.model.cls_token.view(1,c)
        '''
        return x
    def forward(self, x):
        out_feat = self.forward_feat(x)
        #out = self.head(out_feat[:,0])
        torch.cuda.device_of(x)
        return out_feat
    
class AttentionTime(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, Q,K,V):
        B, N, C = Q.shape
        q = self.q(Q).reshape(B, N,  self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(K).reshape(B, N,  self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(V).reshape(B, N,  self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class BlockTime(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1s = norm_layer(dim)
        self.norm1t = norm_layer(dim)
        self.attn = AttentionTime(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, xt,xs):
        nxt = self.norm1t(xt)
        nxs = self.norm1s(xs)
        xst = nxs + self.drop_path(self.attn(nxs,nxt,nxt))
        xst = xst + self.drop_path(self.mlp(self.norm2(xst)))
        return xst
    
class TemporalEncoder(nn.Module):
    def __init__(self, config): 

        super(TemporalEncoder, self).__init__()
        self.feature_size = config['feature_size']
        embed_dim = config['feature_size']
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        act_layer = nn.GELU
        depth = 6
        num_heads = 8
        mlp_ratio=4.
        qkv_bias=True
        drop_rate = 0.
        attn_drop_rate = 0.
        dpr = [x.item() for x in torch.linspace(0, 0., depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            BlockTime(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

    def forward_feat(self, xt,xs):
        
        for layer in self.blocks:
            xt = layer(xt,xs)
            
        x = self.norm(xt)
        return x
    def forward(self, xt,xs):
        out_feat = self.forward_feat(xt,xs)
        #out = self.head(out_feat[:,0])
        #out = self.activation(out)
        torch.cuda.device_of(xt)
        torch.cuda.device_of(xs)
        return out_feat
    
class Model(nn.Module):
    def __init__(self, model, config, nb_class=3,window=10): #nb_class=3; 0 OR 1 OR 2: 0 for no expression ; 1 for MiE ; 2 for MaE 

        super(Model, self).__init__()
        self.img_size = config['img_size']
        self.temp_size = config['temp_size']
        self.feature_size = config['feature_size']
        self.nb_class = nb_class
        self.window = window
        self.embed = self.window//config['temp_size']
        self.SpatialModel = nn.ModuleList([SpatialEncoder(model,config,window=self.window) for _ in range(self.temp_size)])
        self.TemporalModel = nn.ModuleList([TemporalEncoder(config) for _ in range(self.temp_size-1)])
        
        self.start = nn.Linear(config['feature_size']*self.temp_size, 1, bias=True)
        self.Sigmoids = nn.Sigmoid()
        self.length = nn.Linear(config['feature_size']*self.temp_size, 1, bias=True)
        self.Sigmoidl = nn.Sigmoid()
        self.expression = nn.Linear(config['feature_size']*self.temp_size, nb_class, bias=True)
        #self.activation = activation

    def forward_feat(self, x):
        st_features = []
        x1 = x.permute(1,0,2,3,4)
        for i in range(self.temp_size):
            if self.embed == 1:
                input_s = x1[i]
            else:
                input_s = x1[i*self.embed:(i+1)*self.embed]
            if i == 0:
                iet = self.SpatialModel[i](input_s)
                #st_features.append(iet[:,0])
            else:
                ies = self.SpatialModel[i](input_s)
                iet = self.TemporalModel[i-1](iet,ies)
                #st_features.append(ies[:,0])
            st_features.append(iet[:,0])
        return torch.cat(st_features,1)
    
    def forward(self, x):
        out_feat = self.forward_feat(x)
        start = self.Sigmoids(self.start(out_feat))*(self.window-1)
        length = self.Sigmoidl(self.length(out_feat))*(self.window-start)
        expression = self.expression(out_feat)
        #out = self.activation(out)
        torch.cuda.device_of(x)
        return expression, start, length
    
normalize = transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
transforms_test = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "-w",
        "--window",
        default=60,
        type=int,
        help="length of the slidingWindow ",
    )
    
    args = parser.parse_args()
    WIN = args.window
    for name, config in timm_config.items():
        print("#" * 10, name)
        try:
            model = config['load_function'](name, pretrained=True, num_classes=0)
            model = Model(model, config,window=WIN)
            with open(config['path'] + '_config.txt', "a") as f:
                f.write(str(model))
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            test_seq = torch.randn((
                1,
                WIN,
                3,
                config['img_size'][0],
                config['img_size'][1],
            )).to(device)
            print('image_size:  ', config['img_size'])
            print('temp_size:  ', config['temp_size'])
            print('hidden_size: ',
                  model.forward_feat(test_seq).size(1), config['feature_size'])
            print('num_parms:   ', sum(p.numel() for p in model.parameters()))
            torch.save(model, config['path']+'.pth')
            model= torch.load(config['path']+'.pth')
        except:
            pass
