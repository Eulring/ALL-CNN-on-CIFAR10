import warnings

class DefaultConfig(object):
    
    model = 'B'
    checkpoint_load_name = None
    checkpoint_save_name = None
    
    
    stage1 = 100
    stage2 = 100
    stage3 = 50
    stage4 = 50
    
    lr1 = 0.15
    lr2 = 0.1
    lr3 = 0.05
    lr4 = 0.01
    
    test_model_path = '/home/e/Eulring/DL-assignments/midterm-EyEular/ALL-CNN/checkpoints/'
    use_trained_model = False
    
# used for transform learning    
    tl_data_path = '/home/e/Eulring/DL-assignments/midterm-EyEular/ALL-CNN/datasets/class'
    tl_point = None
    class_id = 2
    
    use_clip = True
    clip = 2.0

    
# config of training 1
    batch_size = 128
    use_gpu = True
    num_workers = 4
    print_freq = 5000
    debug_mode = True
    num_train = 49000
    data_aug = True
    #debug_mode = False
    use_cutout = False

# config of training 2
    max_epoch = 20
    lr = 0.01
    lr_decay = 0.9
    weight_decay = 0.001
    

def parse(self, kwargs):
# update the config according to kwargs
    for k, v in kwargs.items():
        if not hasattr(self, k):
            warnings.warn("Warning: opt has not attribut %s" % k)
        setattr(self, k, v)

    print('user config:')
    for k, v in self.__class__.__dict__.items():
        if not k.startswith('__'):
            print(k, getattr(self, k))

DefaultConfig.parse = parse
opt = DefaultConfig()