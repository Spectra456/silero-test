import json 
import torch
import numpy as np
import pandas as pd
from PIL import Image
import torch.utils.data as data
import os
 
class ImageDataset(data.Dataset):

    def __init__(self,dataset_path, filename, original_filename, transform=None):

        df = pd.read_csv(filename)
        df_name = pd.read_csv(original_filename)
        self.df_name = df_name[df_name.columns[0]].to_numpy()

        self.transform = transform
        self.data = df[df.columns[0]].to_numpy()
        self.n_samples = self.data.shape[0]
        self.dataset_path = dataset_path

        
   
    def __len__(self):  
        return self.n_samples
   
    def __getitem__(self, index):
        img = Image.open("{}/{}".format(self.dataset_path, self.data[index]))
        img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return img, self.df_name[index]

