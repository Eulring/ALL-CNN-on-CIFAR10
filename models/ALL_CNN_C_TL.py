import torch
from torch import nn
import torch.nn.functional as F
from .BasicModule import BasicModule


class ALL_CNN_C_TL(BasicModule):
    
    def __init__(self, learned_point, num_classes = 10):
        
        super(ALL_CNN_C_TL, self).__init__()
        
        self.model_name = 'ALL_CNN_C_TL'
        
        self.dp0 = nn.Dropout2d(p = 0.2)
        
        self.conv1 = nn.Conv2d(3, 96, 3, padding = 1)
        #self.bn1 = nn.BatchNorm2d(96)
        
        self.conv2 = nn.Conv2d(96, 96, 3, padding = 1)
        #self.bn2 = nn.BatchNorm2d(96)

        self.conv3 = nn.Conv2d(96, 96, 3, stride = 2, padding = 1)
        self.dp1 = nn.Dropout2d(p = 0.5)
        #self.bn3 = nn.BatchNorm2d(96)
        
        self.conv4 = nn.Conv2d(96, 192, 3, padding = 1)
        #self.bn4 = nn.BatchNorm2d(192)
        
        self.conv5 = nn.Conv2d(192, 192, 3, padding = 1)
        #self.bn5 = nn.BatchNorm2d(192)
        
        self.conv6 = nn.Conv2d(192, 192, 3, stride = 2, padding = 1)
        self.dp2 = nn.Dropout2d(p = 0.5)
        #self.bn6 = nn.BatchNorm2d(192)
        
        
        
        self.conv7 = nn.Conv2d(192, 192, 3, padding = 0)
        #self.bn7 = nn.BatchNorm2d(192)
        
        self.conv8 = nn.Conv2d(192, 192, 1)
        #self.bn8 = nn.BatchNorm2d(192)
        
        self.conv9 = nn.Conv2d(192, 10, 1)
        #self.bn9 = nn.BatchNorm2d(10)
        
        self.avg = nn.AvgPool2d(6)
        
        
        path = './checkpoints/' + learned_point
        #print(path)
        
        
        checkpoint = torch.load(path)
        pretrained_dict = checkpoint['state_dict']
        model_dict = self.state_dict()
        
        overlap_dict = {}
        
        for k, v in pretrained_dict.items():
            if not ('8'in k or '9' in k):
                if k in model_dict:
                    #print(k)
                    overlap_dict[k] = v
                
        model_dict.update(overlap_dict)
        self.load_state_dict(model_dict)
        
        
        
        
        nn.init.xavier_normal_(self.conv8.weight)
        nn.init.xavier_normal_(self.conv9.weight)
        
    def forward(self, x):
        
        x = self.dp0(x)
        
        x = F.relu(self.conv1(x))
        #print(x.shape)
        x = F.relu(self.conv2(x))
        #print(x.shape)
        x = F.relu(self.conv3(x))
        #print(x.shape)
        x = self.dp1(x)
        
        
        x = F.relu(self.conv4(x))
        #print(x.shape)
        x = F.relu(self.conv5(x))
        #print(x.shape)
        x = F.relu(self.conv6(x))
        #print(x.shape
        x = self.dp2(x)
        
        
        x = F.relu(self.conv7(x))
        #print(x.shape)
        x = F.relu(self.conv8(x))
        #print(x.shape)
        x = F.relu(self.conv9(x))
        #x = self.bn3(x)
        #print(x.shape)
        x = self.avg(x)
        x = torch.squeeze(x)
        return x

