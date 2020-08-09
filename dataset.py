import json 
import torch
import numpy as np
import pandas as pd
from PIL import Image
import torch.utils.data as data
import os
 
class ImageDataset(data.Dataset):

    def __init__(self,dataset_path, filename, transform=None):

        df = pd.read_csv(filename)
        
        self.transform = transform
        self.data = df[df.columns[1]].to_numpy()
        self.n_samples = self.data.shape[0]
        self.data = self.data.tolist()
        self.dataset_path = dataset_path
        labels_index = df[df.columns[2]].to_numpy()
        array = np.zeros((len(labels_index), 5), dtype='f')

        for i in range(len(labels_index)):
        	array[i][labels_index[i]] = 1

        self.target = torch.from_numpy(np.array(labels_index)).long()
   
    def __len__(self):  
        return self.n_samples
   
    def __getitem__(self, index):
        img = Image.open("{}/{}".format(self.dataset_path, self.data[index]))
        img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return img, self.target[index]

