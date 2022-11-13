from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import os
import torch
from torchvision import transforms
import numpy as np

class myDataset(Dataset):
    def __init__(self,folder,flag='train'):
        self.folder=folder
        self.flag=flag
        self.read()
    
    def __len__(self):
        return len(self.image)
    
    def read(self):
        imageyuan=np.load(self.folder+'3Dran.npy').squeeze()
        maskyuan=np.load(self.folder+'3Dranm.npy').squeeze()
        
        # imageyuan=np.load(self.folder+'3Dclahe.npy').squeeze()
        # maskyuan=np.load(self.folder+'3Dclahem.npy').squeeze()
        
        rand_i = np.random.choice(range(len(imageyuan)),size=len(imageyuan),replace=False)
        num=len(imageyuan)*0.2
        if self.flag=='train':
            self.image=imageyuan[rand_i[int(num):]]
            self.mask=maskyuan[rand_i[int(num):]]
        elif self.flag=='val':
            self.image=imageyuan[rand_i[:int(num)]]
            self.mask=maskyuan[rand_i[:int(num)]]
    
    def __getitem__(self,index):
        image=self.image[index]
        mask=self.mask[index]
        image=torch.as_tensor(image)
        image=torch.unsqueeze(image,0)
        mask=torch.as_tensor(mask)
        mask=torch.unsqueeze(mask,0)
        return image,mask
        
