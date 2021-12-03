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
#from timm.data import Mixup
#from timm.data.loader import fast_collate
normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

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
class DataLoader(torch.utils.data.Dataset):
    """
    Helper Class to create the pytorch dataset
    """

    def __init__(self, df, transforms=None):
        super().__init__()
        self.img_path = df['url'].values
        self.labels = df['label'].values
        #self.label = df[1:].values
        self.transforms = transforms


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if 'content' not in self.img_path[index]:
            img_path = './content'+self.img_path[index][1:]
        else:
            img_path = './'+self.img_path[index][1:]
        label = self.labels[index]
        img = Image.open(img_path).convert("RGB")
        if self.transforms is not None:
            image = self.transforms(img)
        
        return image, label
    
class fer13_DataLoader(torch.utils.data.Dataset):
    """
    Helper Class to create the pytorch dataset
    """

    def __init__(self, df, transforms=None):
        super().__init__()
        self.pixels = df['pixels'].values
        self.labels = df['emotion'].values
        #self.label = df[1:].values
        self.transforms = transforms


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img_pixels = self.pixels[index]
        label = self.labels[index]
        img_pixels = bytes(int(p) for p in img_pixels.split())
        img = Image.frombuffer('L', (48, 48), img_pixels).convert("RGB")
        if self.transforms is not None:
            image = self.transforms(img)
        
        return image, label

def load_dataset(path, DATA, img_size, batch_size,da,reg):
    #Transforms TRAIN W/O DA_REG
    #random = np.random.choice([
    #                transforms.RandomHorizontalFlip(p=0.5),
    #                transforms.RandomGrayscale(p=0.5),
    #                transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4),
    #                ],1)[0]
    if da and reg:
        transforms_train = transforms.Compose(
            [   
                transforms.Resize((256,256)),
                #transforms.RandomCrop(img_size),
                RandomFiveCrop(img_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomGrayscale(p=0.5),
                transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                #transforms.RandomErasing(scale=(0.02, 0.1)),
                Cutout(n_holes=1, length=16),
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
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
    elif reg:
        transforms_train = transforms.Compose(
            [   
                transforms.Resize((img_size,img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                transforms.RandomErasing(scale=(0.02, 0.1)),
                #Cutout(n_holes=1, length=16),
            ]
        )
    else:
        transforms_train = transforms.Compose(
            [   
                transforms.Resize((img_size,img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
    
    #Transforms VALID
    transforms_valid = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]) 
    
    # READ csv paths
    train_df =  pd.read_csv(path + '/train/'+DATA+'_train.csv') 
    valid_df =  pd.read_csv(path + '/valid/'+DATA+'_valid.csv')
    
    # DATA LOADER
    if DATA =='fer13':
        train_dataset = fer13_DataLoader(
            train_df,
            transforms=transforms_train,
        )
        valid_dataset = fer13_DataLoader(
            valid_df,
            transforms=transforms_valid,
            )
    else:
        train_dataset = DataLoader(
            train_df,
            transforms=transforms_train,
        )
        valid_dataset = DataLoader(
            valid_df,
            transforms=transforms_valid,
            )
    

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        drop_last=True,
        num_workers=2,
        shuffle = True,
        #collate_fn = fast_collate,
    )   
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        drop_last=False,
        num_workers=2,
        
    )
    
    return train_loader, valid_loader, len(train_dataset.labels)


def load_test(path, DATA, img_size, batch_size):

    transforms_test = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    test_df = pd.read_csv(path+'/test/'+DATA+'_test.csv')
    test_dataset = DataLoader(
        test_df,
        transforms=transforms_test,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        drop_last=False,
        num_workers=2,
    )
    return test_loader, len(test_dataset.labels)
