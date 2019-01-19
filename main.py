import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from tqdm import tqdm

import torchvision.datasets as dset
import torchvision.transforms as T
import torch.nn.functional as F 

import numpy as np

from config import opt
from utils import *
import models


def check_acc(loader, model):
    device = torch.device('cuda')
    
    if loader.dataset.train:
        print('**  Checking accuracy on validation set')
    else:
        print('* * * * * *  Checking accuracy on test set')   
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=torch.float32)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('****    Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
    return acc





def train(**kwargs):
    opt.parse(kwargs)
    NUM_TRAIN = 49000

    transform = T.Compose([
                    T.ToTensor(),
                    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ])
    

    
    
    train_transform = T.Compose([])
    if opt.data_aug == True:
        train_transform.transforms.append(T.RandomCrop(32, padding=4))
        train_transform.transforms.append(T.RandomHorizontalFlip())
    train_transform.transforms.append(T.ToTensor())
    train_transform.transforms.append(T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
    if opt.data_aug == True and opt.use_cutout == True:
        train_transform.transforms.append(Cutout(n_holes=1, length=16))


    cifar10_train = dset.CIFAR10('./datasets', train=True, download=True,
                                 transform=train_transform)
    loader_train = DataLoader(cifar10_train, batch_size=opt.batch_size, 
                              sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))

    cifar10_val = dset.CIFAR10('./datasets', train=True, download=True,
                               transform=transform)
    loader_val = DataLoader(cifar10_val, batch_size=opt.batch_size, 
                            sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000)))

    cifar10_test = dset.CIFAR10('./datasets', train=False, download=True, 
                                transform=transform)
    loader_test = DataLoader(cifar10_test, batch_size=opt.batch_size)
    
    
    
    
    
    
    if opt.use_gpu == True:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    model = getattr(models, opt.model)()
    if opt.use_trained_model == True:
        cp_name = opt.model
        if opt.checkpoint_load_name != None:
            cp_name = opt.checkpoint_load_name
        model.load(opt.test_model_path + cp_name)
    model.to(device)
    
    lr = opt.lr
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=opt.weight_decay, nesterov=True)
    
    stages = []
    lrs = []
    if opt.stage1 !=-1:
        stages.append(opt.stage1)
        lrs.append(opt.lr1)
    if opt.stage2 !=-1:
        stages.append(opt.stage2)
        lrs.append(opt.lr2)
    if opt.stage3 !=-1:
        stages.append(opt.stage3)
        lrs.append(opt.lr3)
    if opt.stage4 !=-1:
        stages.append(opt.stage4)
        lrs.append(opt.lr4)
    
    for i, (stage_epoch) in enumerate(stages):
        print(stage_epoch)
        print(lrs[i])
        for param_group in optimizer.param_groups:
            param_group['lr'] = lrs[i]
        
        for epoch in range(stage_epoch):
            for ii, (data, label) in tqdm(enumerate(loader_train)):
                # put data into gpu
                data = data.to(device = device, dtype=torch.float32)
                label = label.to(device = device, dtype=torch.long)

                # get loss
                # print(data.shape)
                scores = model(data)
                loss = F.cross_entropy(scores, label)

                # bp
                optimizer.zero_grad()
                loss.backward()

                if opt.use_clip == True:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip,norm_type=2)

                optimizer.step()

                loss_now = loss.item()

                if ii == 0:
                    print ('Epoch [{}/{}], Loss: {:.4f}, lr :{: f}' 
                        .format(epoch+1, stage_epoch, loss_now, lrs[i]))

            testacc = check_acc(loader_test, model)


            valacc = check_acc(loader_val, model)
            model.update_epoch(lrs[i], loss_now, valacc, testacc)
            model.save(opt.checkpoint_save_name)
            
    

    
    
if __name__ == '__main__':
    import fire
    fire.Fire()
 