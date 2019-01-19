import torch
from torch import nn
import torch.nn.functional as F
from .BasicModule import BasicModule

# python3 main2.py train  --use_trained_model=False --model='B' --checkpoint_save_name='B-1234' --lr1=0.15 --lr2=0.1 --lr3=0.05 --lr4=0.01 --weight_decay=0.0005 --use_clip=True --clip=2.0


# python3 main2.py train --max-epoch=100 --lr=0.1 --use_trained_model=False --lr_decay=1 --model='B' --checkpoint_save_name='B-1'

# python3 main2.py train --max-epoch=100 --lr=0.01 --use_trained_model=True --lr_decay=1 --model='B' --checkpoint_load_name='B-1' --checkpoint_save_name='B-2'

# python3 main2.py train --max-epoch=50 --lr=0.001 --use_trained_model=True --lr_decay=1 --model='B' --checkpoint_load_name='B-2' --checkpoint_save_name='B-3'



# python3 main.py train --max-epoch=50 --lr=0.05  --use_trained_model=True --lr_decay=1  --model='B'
# python3 main.py train --max-epoch=50 --lr=0.01  --use_trained_model=True --lr_decay=0.9  --model='B'

# python3 main.py test  --checkpoint_load_name='B-1' --model='B'

class B(BasicModule):
    
    def __init__(self, num_classes = 10):
        
        super(B, self).__init__()
        
        self.model_name = 'B'
        
        self.dp0 = nn.Dropout2d(p = 0.2)
        
        self.conv1 = nn.Conv2d(3, 96, (5, 5), stride = (1, 1), padding = (2, 2))
        nn.init.xavier_normal_(self.conv1.weight)
        self.conv2 = nn.Conv2d(96, 96, (1, 1), stride = (1, 1), padding = (0, 0))
        nn.init.xavier_normal_(self.conv2.weight)
        self.pool1 = nn.MaxPool2d(kernel_size = 3, stride = 2)
        self.dp1 = nn.Dropout2d(p = 0.5)
        
        self.conv3 = nn.Conv2d(96, 192, (5, 5), stride = (1, 1), padding = (3, 3))
        nn.init.xavier_normal_(self.conv3.weight)
        self.conv4 = nn.Conv2d(192, 192, (1, 1), stride = (1, 1), padding = (0, 0))
        nn.init.xavier_normal_(self.conv4.weight)
        self.pool2 = nn.MaxPool2d(kernel_size = 3, stride = 2)
        self.dp2 = nn.Dropout2d(p = 0.5)
        
        self.conv5 = nn.Conv2d(192, 192, (3, 3), stride = (1, 1), padding = (0, 0))
        nn.init.xavier_normal_(self.conv5.weight)
        
        self.conv6 = nn.Conv2d(192, 192, (1, 1), stride = (1, 1), padding = (0, 0))
        nn.init.xavier_normal_(self.conv6.weight)
        
        self.conv7 = nn.Conv2d(192, 10, (1, 1), stride = (1, 1), padding = (0, 0))
        nn.init.xavier_normal_(self.conv7.weight)
        
        self.avg = nn.AvgPool2d(kernel_size = 6)

        
    def forward(self, x):
        x = self.dp0(x)
        
        x = F.relu(self.conv1(x))
        x = self.pool1(F.relu(self.conv2(x)))
        x = self.dp1(x)

        x = F.relu(self.conv3(x))
        x = self.pool2(F.relu(self.conv4(x)))
        x = self.dp2(x)
        #print(x.shape)
        x = F.relu(self.conv5(x))
        #print(x.shape)
        x = F.relu(self.conv6(x))
        #print(x.shape)
        x = F.relu(self.conv7(x))
        #print(x.shape)
        x = self.avg(x)
        
        x = torch.squeeze(x)
        #assert False
        return x
    








