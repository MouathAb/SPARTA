from pathlib import Path
import numpy as np
import pandas as pd
import torch
import os
from PIL import Image
import torchvision.transforms as transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import numbers
import random 
import glob
import os
from natsort import natsorted
import augmentations

#from timm.data import Mixup
#from timm.data.loader import fast_collate
normalize = transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

class RandomFiveCrop(object):

    def __init__(self, size):
        self.size = size
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            assert len(size) == 2, "Please provide only two dimensions (h, w) for size."
            self.size = size

    def __call__(self, img):
        # randomly return one of the five crops
        return transforms.functional.five_crop(img, self.size)[random.randint(0, 4)]

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)
class AugMixDataset(object):
  """Perform AugMix augmentations and compute mixture.
  Args:
    image: PIL.Image input image
    preprocess: Preprocessing function which should return a torch tensor.
  Returns:
    mixed: Augmented and mixed image.
  """
  def __init__(self,aug_list):
      self.aug_list = aug_list
      self.preprocess = transforms.ToTensor()
      
  def __call__(self,frames):
    ws = np.float32(np.random.dirichlet([1.] * 3))
    m = np.float32(np.random.beta(1., 1.))
    mixed = torch.zeros_like(frames)
    SEED = np.random.randint(0,100)
    for im,image in enumerate(frames):
        mix = torch.zeros_like(image)
        for i in range(3):
            image_aug = transforms.ToPILImage()(image).copy()
            CID = np.random.seed(SEED)
            depth =  np.random.randint(1, 4,)
            SET = np.random.randint(0,100, depth)
            for _ in range(depth):
                CID = np.random.seed(SET[_])
                op = np.random.choice(self.aug_list)
                image_aug = op(image_aug, 1)
                # Preprocessing commutes since all coefficients are convex
            mix += ws[i] * self.preprocess(image_aug)
    
        mixed[im,:,:,:] = (1 - m) * image + m * mix
    return mixed


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img
    
class DataLoaderck(torch.utils.data.Dataset):
    """
    Helper Class to create the pytorch dataset
    """

    def __init__(self, df, transforms=None):
        super().__init__()
        self.img_path = df['url'].values
        self.transforms = transforms


    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, index):
        seq_path = self.img_path[index]
        images_path = sorted(glob.glob(seq_path+'/*'))
        L = len(images_path) # lengeth of the sequence
        if L > 6:
            images_path = images_path[:3] + images_path[L-3:L]
        if L < 6:
            images_path = images_path + [images_path[-1]]*(6-L)
        frames = torch.FloatTensor(6, 3, 224, 224)
        label = torch.LongTensor(6) 
        for i in range(6):
            if i <3:
                label[i] = 0
            else:
                label[i] = 1
        for f,img_path in enumerate(images_path):
            image = Image.open(img_path).convert("RGB")
            if self.transforms is not None:
                image = self.transforms(image)
            else:
                print('check here ! No transforms !')
            frames[f, :, :, :] = image
        
        return frames, label
'''
class DataLoader(torch.utils.data.Dataset):
    """
    Helper Class to create the pytorch dataset
    """

    def __init__(self, df, transforms=None):
        super().__init__()
        self.img_path = df['url'].values
        self.start = df['OnsetFrame'].values
        self.end = df['OffsetFrame'].values
        self.apex = df['ApexFrame'].values
        self.exp = df['Expression'].values
        self.transforms = transforms
        
    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, index):
        global_path = '/mnt/DONNEES/new_ideas_Micro/MERS2/MERS/FER-transformer-master-src'
        seq_path = global_path + self.img_path[index][1:]
        start = self.start[index]
        end = self.end[index]
        exp = self.exp[index]
        apex = self.apex[index]
        
        images_path = natsorted(glob.glob(seq_path+'/*.jpg'))
        if end == 0:
            end = apex + (apex-start) + 1 
        check = end - start  
        if check >= 40:
            if apex - start  >= 20 and end - apex  >= 20:
                if end +10 > len(images_path):
                    if start -10 > 0:
                        images_path =images_path[start-10:start+10] + images_path[apex-10:apex+10] + images_path[end-10:len(images_path)] + [images_path[-1]]*(10-len(images_path)+end)
                    else:
                        images_path = [images_path[0]]*(10-start) + images_path[start-10:start+10] + images_path[apex-10:apex+10] + images_path[end-10:len(images_path)] + [images_path[-1]]*(10-len(images_path)+end)
                    if len(images_path) != 60:
                        print(index,1,1)
                elif start - 10 < 0:
                    if end + 10  < len(images_path):
                        images_path = [images_path[0]]*(10-start) + images_path[0:start+10] + images_path[apex-10:apex+10] + images_path[end-10:end+10]
                    if len(images_path) != 60:
                        print(index,1,2)
                else:
                    images_path = images_path[start-10:start+10] + images_path[apex-10:apex+10] + images_path[end-10:end+10]
                    if len(images_path) != 60:
                        print(index,1,3)
                start = 10
                end = 50
    
            elif end - apex  < 20:
                l = (40 - end + apex )//2
                r = (40 - end + apex )%2
                if start -10 <0:
                    if end +l +r < len(images_path):
                        images_path = [images_path[0]]*(10-start) +images_path[0:start+10] + images_path[apex-l:end+l+r]
                    else:
                        images_path = [images_path[0]]*(10-start) +images_path[0:start+10] + images_path[apex-l:len(images_path)] + (end+l+r-len(images_path))*[images_path[-1]]
                elif end +l+r > len(images_path):
                    if start -10 >0:
                        images_path = images_path[start-10:start+10] + images_path[apex-l:len(images_path)]+ (end+l+r-len(images_path))*[images_path[-1]]
                else:
                    images_path = images_path[start-10:start+10] + images_path[apex-l:end+l+r]
                start = 10
                end = 60 - l - r
    
            else : #elif end - apex >= 20:
                l = (40 - apex + start )//2
                r = (40 -  apex + start )%2
                if start -l -r >0:
                    images_path = images_path[start-l-r:apex+l] + images_path[end-10:end+10]
                else:
                    images_path = [images_path[0]]*(l+r-start) + images_path[0:apex+l] + images_path[end-10:end+10]
                start = l+r
                end = 50
    
        else:
            l = (60 - check)//2
            r = (60 - check)%2
            if start-l-r <0 :
                if end+l < len(images_path):
                    images_path = [images_path[0]]*(l+r-start) +images_path[0:end+l]
                else:
                    images_path = [images_path[0]]*(l+r-start) +images_path[0:len(images_path)] + (end+l-len(images_path))*[images_path[-1]]
            else: 
                if end+l< len(images_path):
                    images_path = images_path[start-l-r:end+l]
                else:
                    images_path = images_path[start-l-r:len(images_path)] + (end+l-len(images_path))*[images_path[-1]]
            start = l+r
            end = 60-l
    
        z = 60
        L = len(images_path) # lengeth of the sequence
        if L != 60:
            print(L)
        assert(L==60)

        frames = torch.FloatTensor(z, 3, 400, 400)
        label = torch.LongTensor(z) 
        for i in range(z):
            if i in range(start,end):
                label[i] = 1
                """
                if exp =='macro-expression':
                    label[i] = 2
                else:
                    label[i] = 1
                """
            else:
                label[i] = 0
        for f,img_path in enumerate(images_path):
            image = Image.open(img_path).convert("RGB")
            frames[f, :, :, :] = transforms.ToTensor()(image)
        if self.transforms is not None:
            frames = self.transforms(frames)
        else:
            print('check here ! No transforms !')
        return frames, label
 
class TestDataLoader(torch.utils.data.Dataset):
    """
    Helper Class to create the pytorch dataset
    """

    def __init__(self, df, transforms=None):
        super().__init__()
        self.img_path = df['url'].values
        self.start = df['OnsetFrame'].values
        self.end = df['OffsetFrame'].values
        self.apex = df['ApexFrame'].values
        self.raw = df['raw'].values
        self.exp = df['Expression'].values
        self.transforms = transforms
        
    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, index):
        global_path = '/mnt/DONNEES/new_ideas_Micro/MERS2/MERS/FER-transformer-master-src'
        seq_path = global_path + self.img_path[index][1:]
        seq_raw_path = global_path + self.raw[index][1:]
        start = self.start[index]
        end = self.end[index]
        exp = self.exp[index]
        apex = self.apex[index]
        
        z = 4374
        
        if end == 0:
            end = apex
        check = end - start + 1 
        
        images_raw_path = natsorted(glob.glob(seq_raw_path+'/*.jpg'))
        L = len(images_raw_path) # lengeth of the sequence
        
        if L < z: #34 is the mean length of [onset,offset] through CAS(ME)2 
            images_raw_path =  images_raw_path + [images_raw_path[-1]]*(z-L)
                
        assert(len(images_raw_path) == z) 
        frames = torch.FloatTensor(z, 3, 480, 640)
        label = torch.LongTensor(z) 
        for i in range(z):
            if i in range(start,end):
                label[i] = 1
                #if exp =='macro-expression':
                #    label[i] = 2
                #else:
                #    label[i] = 1
            else:
                label[i] = 0
        for f,img_path in enumerate(images_raw_path):
            image = Image.open(img_path).convert("RGB")
            frames[f, :, :, :] = transforms.ToTensor()(image)
        if self.transforms is not None:
            frames = self.transforms(frames)
        else:
            print('check here ! No transforms !')
        return frames, label
'''


class DataLoader(torch.utils.data.Dataset):
    """
    Helper Class to create the pytorch dataset
    """

    def __init__(self, df, transforms=None,win=10):
        super().__init__()
        self.img_path = df['url'].values
        self.start = df['OnsetFrame'].values
        self.end = df['OffsetFrame'].values
        self.apex = df['ApexFrame'].values
        self.exp = df['Expression'].values
        self.length = df['length'].values
        self.ind = df['index1'].values
        self.ind2 = df['index2'].values
        self.transforms = transforms
        self.window = win
    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, index):
        global_path = '../'
        seq_path = global_path + self.img_path[index][1:]
        start = self.start[index]
        end = self.end[index]
        exp = self.exp[index]
        apex = self.apex[index]
        ind = self.ind[index]
        ind2 = self.ind2[index]
        length = self.length[index]
        
        w = self.window
        if end == 0:
            end = apex
        check = end - start + 1 
        if ind2 < 1000:
            images_path = natsorted(glob.glob(seq_path+'/*.jpg'))[w*ind2:w*(ind2+1)]
        else:
            images_path = natsorted(glob.glob(seq_path+'/*.jpg'))[length+1000-ind2:]
            images_path = images_path + [images_path[-1]]*(1000+w-ind2)
        L = len(images_path) # lengeth of the sequence
        
        if L != w:
            print(L)
        assert(L==w)

        frames = torch.FloatTensor(w, 3, 400, 400)

        label_start = torch.FloatTensor(1)
        label_start[0] = 0.
        label_length = torch.FloatTensor(1)
        label_length[0] = 0.
        label_exp = torch.LongTensor(1)
        label_exp[0] = 0
        for i in range(w):
            if ind2 < 1000 :
                if i+(ind2*w) == start:
                    label_exp[0] = [j+1 for j,n in enumerate(['micro-expression', 'macro-expression']) if n == exp][0]
                    label_start[0] = i
                    label_length[0] = w - label_start #window -label_start 
                if i+(ind2*w) == end:
                    label_exp[0] = [j+1 for j,n in enumerate(['micro-expression', 'macro-expression']) if n == exp][0]
                    label_length[0] =  i - label_start +1
                
            else: # ind2 = 1000 + length%60 : how mmany frames last for the last window
                if i+length+1000-ind2  == start:
                    label_exp[0] = [j+1 for j,n in enumerate(['micro-expression', 'macro-expression']) if n == exp][0]
                    label_start[0] = i
                    label_length[0] = w - label_start  #window -label_start
                if i+length+1000-ind2  == end:
                    label_exp[0] = [j+1 for j,n in enumerate(['micro-expression', 'macro-expression']) if n == exp][0]
                    label_length[0] = i - label_start +1
        
        label_exp = label_exp.squeeze()
        label_start = label_start.squeeze()
        label_length = label_length.squeeze()
        
        for f,img_path in enumerate(images_path):
            image = Image.open(img_path).convert("RGB")
            frames[f, :, :, :] = transforms.ToTensor()(image)
        if self.transforms is not None:
            frames = self.transforms(frames)
        else:
            print('check here ! No transforms !')
        return frames, (label_exp,label_start,label_length)
    
def slidingwindow(df,win=10):
    df['index1'] = df.index
    df['index2'] = df.index
    df = df.drop(columns='index')
    df2 = pd.DataFrame([],columns=df.columns)
    n =0
    window = win
    for  i in range(len(df)):
        alpha = list(df.loc[i].values)
        for j in range(df.length[i]//window):
            alpha[-1] = j
            df2.loc[n+i,:] = alpha 
            n += 1
            #print(valid_df2)
        if df.length[i]%window != 0:
            alpha[-1] = 1000 + df.length[i]%window # 1000 + how mmany frames last for the last window
            df2.loc[n+i+1,:] = alpha 
            n += 1
    
    df2 = df2.reset_index()
    df2 = df2.drop(columns='index')
    return df2
    
def downsampling(df,keep0s=1000,win=10):
    # downsampling 
           
    the_zeros = []
    the_macros = []
    the_micros = []
    window = win
    for _ in range(len(df)):
        start = df.OnsetFrame.values[_]
        end = df.OffsetFrame.values[_]
        exp = df.Expression.values[_]
        ind2 = df.index2.values[_]
        length = df.length.values[_]
        if ind2 <1000:
            if not(start in range(ind2*window,(ind2+1)*window) or end in range(ind2*window,(ind2+1)*window)):
                the_zeros += [_]
            else:
                if exp == 'macro-expression':
                    the_macros += [_]
                else:
                    the_micros += [_]
        else:
            if not(start in range(length+1000-ind2,length) or end in range(length+1000-ind2,length)):
                the_zeros += [_]
            else:
                if exp == 'macro-expression':
                    the_macros += [_]
                else:
                    the_micros += [_]
    to_keep = sorted(list(np.random.choice(the_zeros,keep0s))+the_macros+the_micros)
    df = df[df.index.isin(to_keep)].reset_index()
    df = df.drop(columns='index')
    return df
    
def load_dataset(path, DATA, img_size, batch_size,da,reg,sub,WIN):
    #Transforms TRAIN W/O DA_REG
    #random = np.random.choice([
    #                transforms.RandomHorizontalFlip(p=0.5),
    #                transforms.RandomGrayscale(p=0.5),
    #                transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4),
    #                ],1)[0]
    aug_list = augmentations.augmentations_all
    if da and reg:
        transforms_train = transforms.Compose(
            [   
                transforms.Resize((256,256)),
                #transforms.RandomCrop(img_size),
                RandomFiveCrop(img_size),
                transforms.RandomHorizontalFlip(p=0.5),
                #transforms.RandomGrayscale(p=0.5),
                transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4),
                #transforms.ToTensor(),
                AugMixDataset(aug_list=aug_list),
                normalize,
                transforms.RandomErasing(scale=(0.02, 0.1)),
                #Cutout(n_holes=1, length=16),
            ]
        )
    elif da:
        transforms_train = transforms.Compose(
            [   
                transforms.Resize((256,256)),
                transforms.RandomCrop(img_size),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomGrayscale(p=0.3),
                transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4),
                #transforms.ToTensor(),
                normalize,
            ]
        )
    elif reg:
        transforms_train = transforms.Compose(
            [   
                transforms.Resize((img_size,img_size)),
                #transforms.ToTensor(),
                normalize,
                transforms.RandomErasing(scale=(0.02, 0.1)),
                #Cutout(n_holes=1, length=16),
            ]
        )
    else:
        transforms_train = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.CenterCrop(img_size),
            #transforms.ToTensor(),
            normalize,    
        ])
    
    #Transforms VALID
    transforms_valid = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.CenterCrop(img_size),
        #transforms.ToTensor(),
        normalize,
    ]) 
    
    # READ csv paths
    df =  pd.read_csv(path + '/train/'+DATA+'_train.csv')
    
    #LOSO && SLIDINGWINDOW 
    vse = [sub] # vse: validation subject
    valid_df = slidingwindow(df[df.Subject.isin(vse)].reset_index(),win=WIN)
    train_df = slidingwindow(df[~df.Subject.isin(vse)].reset_index(),win=WIN)
    
    #DOWNSAMPLING
    train_df = downsampling(train_df,keep0s=1000,win=WIN) # keep0s: how many neutral sequences to kepp after slidingwindow
    

    # DATA LOADER
    train_dataset = DataLoader(
        train_df,
        transforms=transforms_train,
        win=WIN,
        )
    valid_dataset = DataLoader(
        valid_df,
        transforms=transforms_valid,
        win=WIN,
        )
    

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        drop_last=True,
        num_workers=2,
        shuffle=True,
        #collate_fn = fast_collate,
    )   
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        drop_last=False,
        shuffle=False,
        num_workers=2,
        
    )
    
    return train_loader, valid_loader, len(train_dataset.img_path)


def load_test(path, DATA, img_size, batch_size,sub,seq,WIN):

    transforms_test = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.CenterCrop(img_size),
        #transforms.ToTensor(),
        normalize,
    ]) 
    df =  pd.read_csv(path + '/test/'+DATA+'_test.csv') 
    vsu = [sub]
    vse = [seq]
    test_df =  slidingwindow(df[df.Subject.isin(vsu)].reset_index(),win=WIN)
     
    test_df =  test_df[test_df['index1'].isin(vse)].reset_index()

    #try:
    #    test_dataset = TestDataLoader(
    #        test_df,
    #        transforms=transforms_test,
    #    )
    #except:
    test_dataset = DataLoader(
        test_df,
        transforms=transforms_test,
        win=WIN,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        drop_last=False,
        num_workers=2,
    )
    return test_loader, len(test_dataset.img_path)








'''
vse = np.array(['./content/ckckck/ckck/extended-cohn-kanade-images/cohn-kanade-images/S071/002',
       './content/ckckck/ckck/extended-cohn-kanade-images/cohn-kanade-images/S055/003',
       './content/ckckck/ckck/extended-cohn-kanade-images/cohn-kanade-images/S062/002',
       './content/ckckck/ckck/extended-cohn-kanade-images/cohn-kanade-images/S130/001',
       './content/ckckck/ckck/extended-cohn-kanade-images/cohn-kanade-images/S086/001',
       './content/ckckck/ckck/extended-cohn-kanade-images/cohn-kanade-images/S087/001',
       './content/ckckck/ckck/extended-cohn-kanade-images/cohn-kanade-images/S134/002',
       './content/ckckck/ckck/extended-cohn-kanade-images/cohn-kanade-images/S155/002',
       './content/ckckck/ckck/extended-cohn-kanade-images/cohn-kanade-images/S050/001',
       './content/ckckck/ckck/extended-cohn-kanade-images/cohn-kanade-images/S505/006',
       './content/ckckck/ckck/extended-cohn-kanade-images/cohn-kanade-images/S088/002',
       './content/ckckck/ckck/extended-cohn-kanade-images/cohn-kanade-images/S100/006',
       './content/ckckck/ckck/extended-cohn-kanade-images/cohn-kanade-images/S114/003',
       './content/ckckck/ckck/extended-cohn-kanade-images/cohn-kanade-images/S070/002',
       './content/ckckck/ckck/extended-cohn-kanade-images/cohn-kanade-images/S125/001',
       './content/ckckck/ckck/extended-cohn-kanade-images/cohn-kanade-images/S053/001',
       './content/ckckck/ckck/extended-cohn-kanade-images/cohn-kanade-images/S137/003',
       './content/ckckck/ckck/extended-cohn-kanade-images/cohn-kanade-images/S058/001',
       './content/ckckck/ckck/extended-cohn-kanade-images/cohn-kanade-images/S052/002',
       './content/ckckck/ckck/extended-cohn-kanade-images/cohn-kanade-images/S045/003',
       './content/ckckck/ckck/extended-cohn-kanade-images/cohn-kanade-images/S085/002',
       './content/ckckck/ckck/extended-cohn-kanade-images/cohn-kanade-images/S095/003',
       './content/ckckck/ckck/extended-cohn-kanade-images/cohn-kanade-images/S028/001',
       './content/ckckck/ckck/extended-cohn-kanade-images/cohn-kanade-images/S506/006',
       './content/ckckck/ckck/extended-cohn-kanade-images/cohn-kanade-images/S026/006',
       './content/ckckck/ckck/extended-cohn-kanade-images/cohn-kanade-images/S050/006',
       './content/ckckck/ckck/extended-cohn-kanade-images/cohn-kanade-images/S022/003',
       './content/ckckck/ckck/extended-cohn-kanade-images/cohn-kanade-images/S081/003',
       './content/ckckck/ckck/extended-cohn-kanade-images/cohn-kanade-images/S076/006',
       './content/ckckck/ckck/extended-cohn-kanade-images/cohn-kanade-images/S118/003',
       './content/ckckck/ckck/extended-cohn-kanade-images/cohn-kanade-images/S115/004',
       './content/ckckck/ckck/extended-cohn-kanade-images/cohn-kanade-images/S046/003',
       './content/ckckck/ckck/extended-cohn-kanade-images/cohn-kanade-images/S121/005',
       './content/ckckck/ckck/extended-cohn-kanade-images/cohn-kanade-images/S103/001',
       './content/ckckck/ckck/extended-cohn-kanade-images/cohn-kanade-images/S072/006',
       './content/ckckck/ckck/extended-cohn-kanade-images/cohn-kanade-images/S113/007',
       './content/ckckck/ckck/extended-cohn-kanade-images/cohn-kanade-images/S117/005',
       './content/ckckck/ckck/extended-cohn-kanade-images/cohn-kanade-images/S067/005',
       './content/ckckck/ckck/extended-cohn-kanade-images/cohn-kanade-images/S078/004',
       './content/ckckck/ckck/extended-cohn-kanade-images/cohn-kanade-images/S158/002',
       './content/ckckck/ckck/extended-cohn-kanade-images/cohn-kanade-images/S080/005',
       './content/ckckck/ckck/extended-cohn-kanade-images/cohn-kanade-images/S113/005',
       './content/ckckck/ckck/extended-cohn-kanade-images/cohn-kanade-images/S057/006',
       './content/ckckck/ckck/extended-cohn-kanade-images/cohn-kanade-images/S074/005',
       './content/ckckck/ckck/extended-cohn-kanade-images/cohn-kanade-images/S115/002',
       './content/ckckck/ckck/extended-cohn-kanade-images/cohn-kanade-images/S065/003',
       './content/ckckck/ckck/extended-cohn-kanade-images/cohn-kanade-images/S074/005',
       './content/ckckck/ckck/extended-cohn-kanade-images/cohn-kanade-images/S094/004',
       './content/ckckck/ckck/extended-cohn-kanade-images/cohn-kanade-images/S071/004',
       './content/ckckck/ckck/extended-cohn-kanade-images/cohn-kanade-images/S114/006',
       './content/ckckck/ckck/extended-cohn-kanade-images/cohn-kanade-images/S077/005',
       './content/ckckck/ckck/extended-cohn-kanade-images/cohn-kanade-images/S093/004',
       './content/ckckck/ckck/extended-cohn-kanade-images/cohn-kanade-images/S124/002',
       './content/ckckck/ckck/extended-cohn-kanade-images/cohn-kanade-images/S137/002',
       './content/ckckck/ckck/extended-cohn-kanade-images/cohn-kanade-images/S129/012',
       './content/ckckck/ckck/extended-cohn-kanade-images/cohn-kanade-images/S135/001',
       './content/ckckck/ckck/extended-cohn-kanade-images/cohn-kanade-images/S054/002',
       './content/ckckck/ckck/extended-cohn-kanade-images/cohn-kanade-images/S010/002',
       './content/ckckck/ckck/extended-cohn-kanade-images/cohn-kanade-images/S155/002'],
      dtype=object)
'''

'''
df =  pd.read_csv('../data/train/casme2_lv_train.csv')
tdf = df[~df.Subject.isin([1])].reset_index()
global_path = '/home/amouath/MERS/FER-transformer-master-src'
for index,img_path in enumerate(tdf.url.values):
    seq_path = global_path + img_path[1:]
    apex = tdf.ApexFrame.values[index]
    start = tdf.OnsetFrame.values[index]
    end = tdf.OffsetFrame.values[index]
    exp = tdf.Expression.values[index]   
    images_path = natsorted(glob.glob(seq_path+'/*.jpg'))
    if end == 0:
        end = apex + (apex-start) + 1 
    check = end - start  
    if check >= 40:
        if apex - start  >= 20 and end - apex  >= 20:
            if end +10 > len(images_path):
                if start -10 > 0:
                    images_path =images_path[start-10:start+10] + images_path[apex-10:apex+10] + images_path[end-10:len(images_path)] + [images_path[-1]]*(10-len(images_path)+end)
                else:
                    images_path = [images_path[0]]*(10-start) + images_path[start-10:start+10] + images_path[apex-10:apex+10] + images_path[end-10:len(images_path)] + [images_path[-1]]*(10-len(images_path)+end)
                if len(images_path) != 60:
                    print(index,1,1)
            elif start - 10 < 0:
                if end + 10  < len(images_path):
                    images_path = [images_path[0]]*(10-start) + images_path[0:start+10] + images_path[apex-10:apex+10] + images_path[end-10:end+10]
                if len(images_path) != 60:
                    print(index,1,2)
            else:
                images_path = images_path[start-10:start+10] + images_path[apex-10:apex+10] + images_path[end-10:end+10]
                if len(images_path) != 60:
                    print(index,1,3)
            start = 10
            end = 50

        elif end - apex  < 20:
            l = (40 - end + apex )//2
            r = (40 - end + apex )%2
            if start -10 <0:
                if end +l +r < len(images_path):
                    images_path = [images_path[0]]*(10-start) +images_path[0:start+10] + images_path[apex-l:end+l+r]
                else:
                    images_path = [images_path[0]]*(10-start) +images_path[0:start+10] + images_path[apex-l:len(images_path)] + (end+l+r-len(images_path))*[images_path[-1]]
            elif end +l+r > len(images_path):
                if start -10 >0:
                    images_path = images_path[start-10:start+10] + images_path[apex-l:len(images_path)]+ (end+l+r-len(images_path))*[images_path[-1]]
            else:
                images_path = images_path[start-10:start+10] + images_path[apex-l:end+l+r]
            start = 10
            end = 60 - l - r

        else : #elif end - apex >= 20:
            l = (40 - apex + start )//2
            r = (40 -  apex + start )%2
            if start -l -r >0:
                images_path = images_path[start-l-r:apex+l] + images_path[end-10:end+10]
            else:
                images_path = [images_path[0]]*(l+r-start) + images_path[0:apex+l] + images_path[end-10:end+10]
            start = l+r
            end = 50

    else:
        l = (60 - check)//2
        r = (60 - check)%2
        if start-l-r <0 :
            if end+l < len(images_path):
                images_path = [images_path[0]]*(l+r-start) +images_path[0:end+l]
            else:
                images_path = [images_path[0]]*(l+r-start) +images_path[0:len(images_path)] + (end+l-len(images_path))*[images_path[-1]]
        else: 
            if end+l< len(images_path):
                images_path = images_path[start-l-r:end+l]
            else:
                images_path = images_path[start-l-r:len(images_path)] + (end+l-len(images_path))*[images_path[-1]]
        start = l+r
        end = 60-l

    z = 60
    assert(len(images_path)==60)
''' 
