import torch
import torch.nn as nn
import torch.nn.functional as F
from .BasicModule import BasicModule

# python3 main2.py train  --use_trained_model=False --model='ConvPool_CNN_C_1' --checkpoint_save_name='ConvPool_CNN_C_1-1' --lr1=0.1 --lr2=0.1 --lr3=0.05 --lr4=0.01 --weight_decay=0.0001 --use_clip=False

    
class ConvPool_CNN_C_1(BasicModule):

    def __init__(self, num_classes=10):
        super (ConvPool_CNN_C_1, self).__init__ ()
        
        self.model_name = 'ConvPool_CNN_C_1'
        
        input_channel = 3
        
        use_dropout=True
        use_bn=False
        
        layers=[]
        layers.append(nn.Conv2d(input_channel, 96, 3, padding = 1))
        if use_bn:
            layers.append(nn.BatchNorm2d(96))
        layers.append(nn.ReLU (inplace=True))

        layers.append(nn.Conv2d(96, 96, 3, padding = 1))
        if use_bn:
            layers.append(nn.BatchNorm2d(96))
        layers.append(nn.ReLU (inplace=True))

        layers.append(nn.Conv2d(96, 96, 3, padding = 1))
        if use_bn:
            layers.append(nn.BatchNorm2d(96))
        layers.append(nn.ReLU (inplace=True))

        layers.append(nn.MaxPool2d(3,stride = 2)) # (N,96,16,16))
        if use_dropout:
            layers.append(nn.Dropout2d(0.5))

        layers.append (nn.Conv2d(96, 192, 3, padding = 1))
        if use_bn:
            layers.append (nn.BatchNorm2d (192))
        layers.append (nn.ReLU (inplace=True))

        layers.append (nn.Conv2d(192, 192, 3, padding = 1))
        if use_bn:
            layers.append (nn.BatchNorm2d (192))
        layers.append (nn.ReLU (inplace=True))

        layers.append (nn.Conv2d(192, 192, 3, padding = 2))
        if use_bn:
            layers.append (nn.BatchNorm2d (192))
        layers.append (nn.ReLU (inplace=True))

        layers.append (nn.MaxPool2d(3,stride = 2))  # (N,96,8,8))
        if use_dropout:
            layers.append (nn.Dropout2d(0.5))
            
            
        layers.append (nn.Conv2d(192, 192, 3, stride=1, padding=0, bias=True))    
        if use_bn:
            layers.append (nn.BatchNorm2d (192))
        layers.append (nn.ReLU (inplace=True))
        
        layers.append (nn.Conv2d(192, 192, 3, stride=1, padding=0, bias=True))    
        if use_bn:
            layers.append (nn.BatchNorm2d (192))
        layers.append (nn.ReLU (inplace=True))
            
        layers.append (nn.Conv2d(192, 10, 3, stride=1, padding=0, bias=True))    
        if use_bn:
            layers.append (nn.BatchNorm2d (10))
        layers.append (nn.ReLU (inplace=True))
            
        
        layers.append (nn.AdaptiveAvgPool2d(1))
        

        
        self.layer = nn.Sequential(*layers)
   
        
        
        for m in self.modules():
            name = m.__class__.__name__
            if name.find is 'Conv':
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

                


    def forward(self, x):
        out = self.layer(x)
        out = out.view(-1, 10)
        # print(out.shape)
        
        return out