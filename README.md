# ALL-CNN-on-CIFAR10

Pytorch Implementation of ALL-CNN in CIFAR10 Dataset 

the model can reach 93% accuracy in CIFAR-10 dataset 


## Setting in Config

__model__: select the model to train.

__state1-4__: the experient in paper using 4 traing stage with different learning-rate.

__use_clip__: set ture to avoid the gradient explodsion.

__data_aug__: using crop and flip of images to augment the data.

__use_cutout__: using cutout[https://github.com/uoguelph-mlrg/Cutout] to augment the data.



--- 

## Command

use the command below to train the model:

### ALL-CNN-C model

> python3 main.py train  --use_trained_model=False  --model='ALL_CNN_C' --lr1=0.1 --lr2=0.05 --lr3=0.01 --lr4=0.001 --weight_decay=0.0001 --use_clip=True --clip=2.0 --use_cutout=False --data_aug=False --class_id=1 --checkpoint_save_name='ALL_CNN_C'

### ALL-CNN-C model (with data augmentation)

> python3 main.py train  --use_trained_model=False  --model='ALL_CNN_C' --lr1=0.1 --lr2=0.05 --lr3=0.01 --lr4=0.001 --weight_decay=0.0005 --use_clip=True --clip=2.0 --use_cutout=True --data_aug=True --checkpoint_save_name='ALL_CNN_C_aug'


