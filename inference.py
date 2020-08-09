import os
import sys
import torch
import argparse
import datetime
from tqdm import tqdm
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
from dataset_inference import *
from model import *

array_df = []
classes = ['speech', 'music', 'noise']

def test(device, loader):
    with torch.no_grad():
        
        progress_bar = tqdm(loader)

        for (data,path) in progress_bar:
            progress_bar.set_description('Test ')
            data = data.to(device)
            output = model(data)
            array_df.append([path[0],np.argmax(output.cpu().detach().numpy())])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1)  
    parser.add_argument('--batch_size', type=int, default=1) 
    parser.add_argument('--img_size', type=tuple, default=(128, 128), help='input size of image ')
    parser.add_argument('--weights', type=str, default='weights', help='path for saving weight')
    parser.add_argument('--num_workers', type=int, default=os.cpu_count(), help='number of threads for data loader, by default using all cores')
    parser.add_argument('--dataset', type=str, default='dataset', help='path of dataset')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for optimizer')
    parser.add_argument('--device', type=str, default='cuda', help='Device for training')

    args = parser.parse_args()

    size = args.img_size
    batch_size = args.batch_size
    num_workers = args.num_workers
    epochs = 1
    path = args.dataset
    learning_rate = args.learning_rate
    device = torch.device(args.device)


    test_transformations = transforms.Compose([transforms.Resize((size[0],size[1])),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    test_dataset = ImageDataset('/submission_jpg','/sample_submission_inf.csv','sample_submission.csv', transform=test_transformations)

    test_loader = DataLoader(test_dataset,
                              batch_size=batch_size,
                              num_workers=num_workers,
                             )


    model = ImageClassifier(size).to(device)
    checkpoint = torch.load('weights/best_loss1.737_accuracy0.944.pt')
    model.load_state_dict(checkpoint)
    model.eval()

    try:
        for epoch in range(0, epochs):
            test_loss, test_accuracy = test(device, test_loader)
    except:
        pass
    
    df = pd.DataFrame(data=array_df,columns=['wav_path','target'])
    df.to_csv('my_subm.csv',index=False)
