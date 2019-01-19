import torch
from torch import nn
import torch.nn.functional as F
from .BasicModule import BasicModule
import numpy as np

# python3 main2.py train  --use_trained_model=False --model='A' --checkpoint_save_name='A-1234' --lr1=0.15 --lr2=0.1 --lr3=0.05 --lr4=0.01 --weight_decay=0.0005 --use_clip=True --clip=1.0

# python3 main2.py train --max-epoch=100 --lr=0.1 --use_trained_model=False --lr_decay=1 --model='A' --checkpoint_save_name='A-1'

# python3 main2.py train --max-epoch=100 --lr=0.01 --use_trained_model=True --lr_decay=1 --model='A' --checkpoint_load_name='A-1' --checkpoint_save_name='A-2'

# python3 main2.py train --max-epoch=50 --lr=0.001 --use_trained_model=True --lr_decay=1 --model='A' --checkpoint_load_name='A-2' --checkpoint_save_name='A-3'

# python3 main.py train --max-epoch=50 --lr=0.001  --use_trained_model=True --lr_decay=1  --model='A' --checkpoint_load_name='A-3' --checkpoint_save_name='A-4'

# python3 main.py train --max-epoch=50 --lr=0.0005  --use_trained_model=True --lr_decay=0.9  --model='A-2'
# python3 main.py test  --checkpoint_load_name='A-3' --model='A'

## add batch normalization

class A(BasicModule):
    
    def __init__(self, num_classes = 10):
        
        super(A, self).__init__()
        
        self.model_name = 'A'
        
        self.dp0 = nn.Dropout2d(p = 0.2)
        
        self.conv1 = nn.Conv2d(3, 96, (5, 5), stride = (1, 1), padding = (2, 2))
        nn.init.xavier_normal_(self.conv1.weight)
        self.pool1 = nn.MaxPool2d(kernel_size = 3, stride = 2)
        self.dp1 = nn.Dropout2d(p = 0.5)
        #self.bn1 = nn.BatchNorm2d(96)
        
        self.conv2 = nn.Conv2d(96, 192, (5, 5), stride = (1, 1), padding = (3, 3))
        nn.init.xavier_normal_(self.conv2.weight)
        self.pool2 = nn.MaxPool2d(kernel_size = 3, stride = 2)
        self.dp2 = nn.Dropout2d(p = 0.5)
        #self.bn2 = nn.BatchNorm2d(192)
        
        self.conv3 = nn.Conv2d(192, 192, (3, 3), stride = (1, 1), padding = (0, 0))
        nn.init.xavier_normal_(self.conv3.weight)
        #self.bn3 = nn.BatchNorm2d(192)
        
        self.conv4 = nn.Conv2d(192, 192, (1, 1), stride = (1, 1), padding = (0, 0))
        nn.init.xavier_normal_(self.conv4.weight)
        #self.bn4 = nn.BatchNorm2d(192)
        
        self.conv5 = nn.Conv2d(192, 10, (1, 1), stride = (1, 1), padding = (0, 0))
        nn.init.xavier_normal_(self.conv5.weight)
        #self.bn5 = nn.BatchNorm2d(10)
        
        self.avg = nn.AvgPool2d(kernel_size = 6)
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
        '''
        
    def forward(self, x): 
        x = self.dp0(x)
        
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.dp1(x) 
        
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.dp2(x) 
        
        x = F.relu(self.conv3(x))
        #print(x.shape)
        x = F.relu(self.conv4(x))
        #print(x.shape)
        x = F.relu(self.conv5(x))
        #print(x.shape)
        x = self.avg(x)
        x = x.view(-1, 10)
        #print(x.shape)
        
        return x
