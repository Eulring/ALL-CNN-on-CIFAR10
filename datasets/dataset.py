import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T

class CIFAR100(data.Dataset):
    
    def __init__(self, data_path, split='train', transform=None):
        self.split = split
        if self.split == 'test':
            data_path += '/test'
        else:
            data_path += '/train'
        
        print(data_path)
        import pickle
        with open(data_path, 'rb') as fo:
            dicts = pickle.load(fo, encoding='latin1')
            
        self.imgs = []
        self.labels = []
        
        # The imgs size (5000, 3070)
        self.imgs.append(dicts['data'])
        self.labels = dicts['fine_labels']
        
        # The imgs size (5000, 3, 32, 32)
        self.imgs = np.vstack(self.imgs).reshape(-1, 3, 32, 32)
        self.imgs = self.imgs.transpose((0, 2, 3, 1))
        
        print(self.imgs.shape)
        print(len(self.labels))
        
        if self.split == 'train':
            self.imgs = self.imgs[:4500, :]
            self.labels = self.labels[:4500]
        elif self.split == 'val':
            self.imgs = self.imgs[4500:, :]
            self.labels = self.labels[4500:]
            
        if transform == None:
            self.transform = T.Compose([
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        else:
            self.transform = transform
            

        
    def __getitem__(self, index):
        # The imgs size (3, 32, 32)
        img = self.imgs[index, :]
        
        # The imgs size (32, 32, 3) is converted by Image.fromarray
        img = Image.fromarray(np.uint8(img))
        
        # The imgs size (3, 32, 32) is converted by ToTensor()
        img = self.transform(img)
        
        #print(img.shape)
        return img, self.labels[index]
        
        
    def __len__(self):
        return self.imgs.shape[0]
        
        

