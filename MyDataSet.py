import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import ndimage

class MyDataset(Dataset):
    
    def __init__(self, datas=None, labels=None, shape=None, input_D=None, input_H=None, input_W=None, phase='train', transforms=None):
        self.datas = datas
        self.labels = labels
        self.transforms = transforms
        self.shape = shape
        self.input_D = input_D
        self.input_H = input_H
        self.input_W = input_W
        self.phase = phase

    #返回整个数据集大小
    def __len__(self):
        return self.datas.shape[0]
    
    #根据索引index返回dataset[index]
    def __getitem__(self,index):
        if self.phase == 'train':
            img = self.__data_process__(self.datas[index])
            label = self.labels[index]
            img = torch.tensor(img)
            if self.transforms:
                img = self.transforms(img)
            return img,label
        elif self.phase == 'test':
            img = self.__data_process__(self.datas[index])
            img = torch.tensor(img)
            if self.transforms:
                img = self.transforms(img)
            return img
    
    def __itensity_normalize_one_volume__(self, volume):
        '''
        normalize the itensity of an nd volume based on the mean and std of nonzeor region
        inputs:
            volume: the input nd volume
        outputs:
            out: the normalized nd volume
        '''
        
        pixels = volume[volume > 0]
        mean = pixels.mean()
        std  = pixels.std()
        out = (volume - mean)/std
        out_random = np.random.normal(0, 1, size = volume.shape)
        out[volume == 0] = out_random[volume == 0]
        return out

    def __resize_data__(self, data):
        '''
        Resize the data to the input size
        ''' 
        if self.shape == 2:
            [depth, height, width] = data.shape
            scale = [self.input_D*1.0/depth, self.input_H*1.0/height, self.input_W*1.0/width]  
        else:
            [channel, depth, height, width] = data.shape
            scale = [channel,self.input_D*1.0/depth, self.input_H*1.0/height, self.input_W*1.0/width]  
        data = ndimage.interpolation.zoom(data, scale, order=0)

        return data
    
    def __data_process__(self, data): 

        # resize data
        data = self.__resize_data__(data)

        # normalization datas
        data = self.__itensity_normalize_one_volume__(data)

        return data