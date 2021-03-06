from torch import nn, sigmoid, softmax 
import torch.nn.functional as F


class ImageClassifier(nn.Module):

    def __init__(self, size):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding = 1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding= 1)
        self.conv2_drop = nn.Dropout2d()
        input_size = int((128*(size[0]/4)*(size[1]/4))) # Getting size of input image from init for calculate first linear layer size 
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 3)
 
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return softmax(x, dim=1)
