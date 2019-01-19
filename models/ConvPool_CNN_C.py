import torch
from torch import nn
import torch.nn.functional as F
from .BasicModule import BasicModule
import numpy as np

# python3 main2.py train  --use_trained_model=False --model='ConvPool_CNN_C' --checkpoint_save_name='ConvPool_CNN_C-1234' --lr1=0.15 --lr2=0.1 --lr3=0.05 --lr4=0.01 --weight_decay=0.001 --use_clip=True --clip=2.0


# python3 main2.py train --stage1=3 --use_trained_model=False --model='ConvPool_CNN_C' --checkpoint_save_name='ConvPool_CNN_C-1234'

# python3 main2.py train --max-epoch=100 --lr=0.02 --use_trained_model=True --lr_decay=1 --model='ConvPool_CNN_C' --checkpoint_load_name='ConvPool_CNN_C-1' --checkpoint_save_name='ConvPool_CNN_C-2'

# python3 main2.py train --max-epoch=50 --lr=0.001 --use_trained_model=True --lr_decay=1 --model='ConvPool_CNN_C' --checkpoint_load_name='ConvPool_CNN_C-2' --checkpoint_save_name='ConvPool_CNN_C-3'



class ConvPool_CNN_C(BasicModule):
    
    def __init__(self, num_classes = 10):
        
        super(ConvPool_CNN_C, self).__init__()
        
        self.model_name = 'ConvPool_CNN_C'
        
        self.dp0 = nn.Dropout2d(p = 0.2)
        
        #self.conv1 = nn.Conv2d(3, 96, 3)
        self.conv1 = nn.Conv2d(3, 96, (3, 3), stride = (1, 1), padding = (1, 1))
        nn.init.xavier_normal_(self.conv1.weight)
        #self.bn1 = nn.BatchNorm2d(96)
        
        #self.conv2 = nn.Conv2d(96, 96, 3, 1, 1)
        self.conv2 = nn.Conv2d(96, 96, (3, 3), stride = (1, 1), padding = (1, 1))
        nn.init.xavier_normal_(self.conv2.weight)
        #self.bn2 = nn.BatchNorm2d(96)
        
        #self.conv3 = nn.Conv2d(96, 96, 3, 2, 2)
        self.conv3 = nn.Conv2d(96, 96, (3, 3), stride = (1, 1), padding = (1, 1))
        nn.init.xavier_normal_(self.conv3.weight)
        self.pool1 = nn.MaxPool2d((3, 3), stride = (2, 2))
        self.dp1 = nn.Dropout2d(p = 0.5)
        #self.bn3 = nn.BatchNorm2d(96)
        
        
        #self.conv4 = nn.Conv2d(96, 192, 3, 1, 1)
        self.conv4 = nn.Conv2d(96, 192, (3, 3), stride = (1, 1), padding = (1, 1))
        nn.init.xavier_normal_(self.conv4.weight)
        #self.bn4 = nn.BatchNorm2d(192)
        
        self.conv5 = nn.Conv2d(192, 192, (3, 3), stride = (1, 1), padding = (1, 1))
        nn.init.xavier_normal_(self.conv5.weight)
        #self.bn5 = nn.BatchNorm2d(192)
        
        self.conv6 = nn.Conv2d(192, 192, (3, 3), stride = (1, 1), padding = (0, 0))
        nn.init.xavier_normal_(self.conv6.weight)
        self.pool2 = nn.MaxPool2d((3, 3), stride = (2, 2))
        self.dp2 = nn.Dropout2d(p = 0.5)
        #self.bn6 = nn.BatchNorm2d(192)
        
        
        self.conv7 = nn.Conv2d(192, 192, (3, 3), stride = (1, 1), padding = (1, 1))
        nn.init.xavier_normal_(self.conv7.weight)
        #self.bn7 = nn.BatchNorm2d(192)
        
        self.conv8 = nn.Conv2d(192, 192, (1, 1), stride = (1, 1))
        nn.init.xavier_normal_(self.conv8.weight)
        #self.bn8 = nn.BatchNorm2d(192)
        
        self.conv9 = nn.Conv2d(192, 10, (1, 1), stride = (1, 1))
        nn.init.xavier_normal_(self.conv9.weight)
        #self.bn9 = nn.BatchNorm2d(10)

        
        self.avg = nn.AvgPool2d(6)
        
    def forward(self, x):
        
        x = self.dp0(x)
        
        x = F.relu(self.conv1(x))
        #print(x.shape)
        x = F.relu(self.conv2(x))
        #print(x.shape)
        x = F.relu(self.conv3(x))
        x = self.pool1(x)
        x = self.dp1(x)
        
        
        x = F.relu(self.conv4(x))
        #print(x.shape)
        x = F.relu(self.conv5(x))
        #print(x.shape)
        x = F.relu(self.conv6(x))
        #print(x.shape)
        x = self.pool2(x)
        x = self.dp2(x)
        
        #print(x.shape)
        x = F.relu(self.conv7(x))
        #print(x.shape)
        x = F.relu(self.conv8(x))
        #print(x.shape)
        x = F.relu(self.conv9(x))
        #x = self.bn3(x)
        #print(x.shape)
        x = self.avg(x)
        x = x.view(-1, 10)
        return x

