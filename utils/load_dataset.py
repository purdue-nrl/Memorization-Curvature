import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data.distributed import DistributedSampler
from wilds import get_dataset
import os
import logging

class Dict_To_Obj:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def get_transform(
    test_transform,
    train_transform,
    val_transform,
    mean,
    std,
    augment,
    img_dim,
    padding_crop,
    resize=False):

    if(test_transform == None):
        test_transform = transforms.Compose([
                                                transforms.Resize((img_dim, img_dim)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean, std),
                                            ])

    if(val_transform == None):    
        val_transform = test_transform

    if(train_transform == None):
        if augment:
            transforms_list = [
                transforms.Resize((img_dim, img_dim)),
                transforms.RandomCrop(img_dim, padding=padding_crop),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
            if resize:
                transforms_list = [transforms.Resize((img_dim, img_dim))] + transforms_list
                train_transform = transforms.Compose(transforms_list)
            else:
                train_transform = transforms.Compose(transforms_list)
        else:
            train_transform = test_transform

    return train_transform, val_transform, test_transform

def load_dataset(
    dataset='CIFAR10',
    train_batch_size=128,
    test_batch_size=128,
    val_split=0.0,
    augment=True,
    padding_crop=4,
    num_workers=None,
    shuffle=True,
    random_seed=None,
    resize_shape=None,
    mean=None,
    std=None,
    train_transform=None,
    test_transform=None,
    val_transform=None,
    index=None,
    root_path=None,
    logger=None,
    distributed=False):
    '''
    Inputs
    dataset -> CIFAR10, CIFAR100, TinyImageNet, ImageNet
    train_batch_size -> batch size for training dataset
    test_batch_size -> batch size for testing dataset
    val_split -> percentage of training data split as validation dataset
    augment -> bool flag for Random horizontal flip and shift with padding
    padding_crop -> units of pixel shift
    shuffle -> bool flag for shuffling the training and testing dataset
    random_seed -> fixes the shuffle seed for reproducing the results
    return -> bool for returning the mean, std, img_size
    '''
    # Load dataset
    # Use the following transform for training and testing

    args_num_workers = num_workers

    if logger is None:
        logger = logging.getLogger(f'Default logger')
        logger.setLevel(logging.INFO)

    if (dataset.lower() == 'mnist'):
        if(mean == None):
            mean = [0.1307]
        if(std == None):
            std = [0.3081]
        img_dim = 28
        img_ch = 1
        num_classes = 10
        num_worker = 2
        root = os.path.join(root_path, 'mnist')

        train_transform, val_transform, test_transform = get_transform(
            test_transform,
            train_transform,
            val_transform,
            mean,
            std,
            augment,
            img_dim,
            padding_crop)

        trainset = torchvision.datasets.MNIST(
            root=root,
            train=True,
            download=True,
            transform=train_transform)

        valset = torchvision.datasets.MNIST(
            root=root,
            train=True,
            download=True,
            transform=val_transform)

        testset = torchvision.datasets.MNIST(
            root=root,
            train=False,
            download=True,
            transform=test_transform)
        
    elif(dataset.lower() == 'cifar10'):
        if(mean == None):
            mean = [0.4914, 0.4822, 0.4465]
        if(std == None):
            std = [0.2023, 0.1994, 0.2010]

        img_dim = 32
        img_ch = 3
        num_classes = 10
        num_worker = 4
        root = os.path.join(root_path, 'CIFAR10')

        train_transform, val_transform, test_transform = get_transform(
            test_transform,
            train_transform,
            val_transform,
            mean,
            std,
            augment,
            img_dim,
            padding_crop)

        trainset = torchvision.datasets.CIFAR10(
            root=root,
            train=True,
            download=True,
            transform=train_transform)

        valset = torchvision.datasets.CIFAR10(
            root=root,
            train=True,
            download=True,
            transform=val_transform)

        testset = torchvision.datasets.CIFAR10(
            root=root,
            train=False,
            download=True,
            transform=test_transform)

    # http://cs231n.stanford.edu/tiny-imagenet-200.zip
    elif(dataset.lower() == 'tinyimagenet' or dataset.lower() == 'tin'):
        if(mean == None):
            mean = [0.485, 0.456, 0.406]
        if(std == None):
            std = [0.229, 0.224, 0.225]

        root = os.path.join(root_path, 'TinyImageNet')
        img_dim = 64
        resize = False
        if(resize_shape == None):
            resize_shape = (32, 32)
            img_dim = 32
        else:
            img_dim = resize_shape[0]
            resize = True
        img_ch = 3
        num_classes = 200
        num_worker = 4
        
        train_transform, val_transform, test_transform = get_transform(
            test_transform,
            train_transform,
            val_transform,
            mean,
            std,
            augment,
            img_dim,
            padding_crop,
            resize=resize)

        trainset = TinyImageNet(root=root, transform=test_transform, train=True) 
        valset = TinyImageNet(root=root, transform=test_transform, train=True) 
        testset = TinyImageNet(root=root, transform=test_transform, train=False)
  
    elif(dataset.lower() == 'svhn'):
        if(mean == None):
            mean = [0.4376821,  0.4437697,  0.47280442]
        if(std == None):
            std = [0.19803012, 0.20101562, 0.19703614]
        img_dim = 32
        img_ch = 3
        num_classes = 10
        num_worker = 2
        root = os.path.join(root_path, 'SVHN')

        train_transform, val_transform, test_transform = get_transform(
            test_transform,
            train_transform,
            val_transform,
            mean,
            std,
            augment,
            img_dim,
            padding_crop)

        trainset = torchvision.datasets.SVHN(
            root=root,
            split='train',
            download=True,
            transform=train_transform)

        valset = torchvision.datasets.SVHN(
            root=root,
            split='train',
            download=True,
            transform=val_transform)

        testset = torchvision.datasets.SVHN(
            root=root,
            split='test',
            download=True, 
            transform=test_transform)

    elif(dataset.lower() == 'lsun'):
        if(mean == None):
            mean = [0.5071, 0.4699, 0.4326]
        if(std == None):
            std = [0.2485, 0.2492, 0.2673]
        img_dim = 32
        resize = False
        if(resize_shape == None):
            resize_shape = (img_dim, img_dim)
        else:
            img_dim = resize_shape[0]
            resize = True
        img_ch = 3
        num_classes = 10
        num_worker = 4        
       
        train_transform, val_transform, test_transform = get_transform(
            test_transform,
            train_transform,
            val_transform,
            mean,
            std,
            augment,
            img_dim,
            padding_crop,
            resize=resize)

        root = os.path.join(root_path, 'LSUN')
        trainset = torchvision.datasets.LSUN(
            root=root,
            classes='train',
            transform=train_transform)

        valset = torchvision.datasets.LSUN(
            root=root,
            classes='val',
            transform=val_transform)

        testset = torchvision.datasets.LSUN(
            root=root,
            classes='val',
            transform=test_transform)

    elif(dataset.lower() == 'places365'):
        if(mean == None):
            mean = [0.4578, 0.4413, 0.4078]
        if(std == None):
            std = [0.2435, 0.2418, 0.2622]
        
        img_dim = 32
        if(resize_shape == None):
            resize_shape = (img_dim, img_dim)
        img_ch = 3
        num_classes = 365
        num_worker = 4

        train_transform, val_transform, test_transform = get_transform(
            test_transform,
            train_transform,
            val_transform,
            mean,
            std,
            augment,
            img_dim,
            padding_crop)

        root = os.path.join(root_path, 'Places365')
        trainset = torchvision.datasets.Places365(
            root=root,
            split='train-standard',
            transform=train_transform,
            download=False)

        valset = torchvision.datasets.Places365(
            root=root,
            split='train-standard',
            transform=val_transform,
            download=False)

        testset = torchvision.datasets.Places365(
            root=root,
            split='val',
            transform=test_transform,
            download=False)

    elif(dataset.lower() == 'cifar100'):
        if(mean == None):
            mean = [0.5071, 0.4867, 0.4408]
        if(std == None):
            std = [0.2675, 0.2565, 0.2761]

        img_dim = 32
        img_ch = 3
        num_classes = 100
        num_worker = 4
        root = os.path.join(root_path, 'CIFAR100')

        train_transform, val_transform, test_transform = get_transform(
            test_transform,
            train_transform,
            val_transform,
            mean,
            std,
            augment,
            img_dim,
            padding_crop)

        trainset = torchvision.datasets.CIFAR100(
            root=root,
            train=True,
            download=True,
            transform=train_transform)

        valset = torchvision.datasets.CIFAR100(
            root=root,
            train=True,
            download=True,
            transform=val_transform)

        testset = torchvision.datasets.CIFAR100(
            root=root,
            train=False,
            download=True,
            transform=test_transform)  

    elif(dataset.lower() in 'textures'):
        if(mean == None or std == None):
            raise ValueError("Mean and std for textures no supported yet!")
        img_dim = 32
        if(resize_shape == None):
            resize_shape = (img_dim, img_dim)
        img_ch = 3
        num_classes = 47
        num_worker = 4
        root = os.path.join(root_path, 'Textures')
                
        train_transform, val_transform, test_transform = get_transform(
            test_transform,
            train_transform,
            val_transform,
            mean,
            std,
            augment,
            img_dim,
            padding_crop)

        trainset = torchvision.datasets.ImageFolder(
            root=os.path.join(root,'images'),
            transform=train_transform)

        valset = torchvision.datasets.ImageFolder(
            root=os.path.join(root,'images'),
            transform=val_transform)

        testset = torchvision.datasets.ImageFolder(
            root=os.path.join(root,'images'),
            transform=test_transform)

    elif(dataset.lower() == 'u-noise'):
        img_dim = 32
        img_ch = 3
        num_classes = 1
        num_worker = 4

        if(mean == None or std == None):
            raise ValueError("Mean and std for textures no supported yet!")
                
        train_transform, val_transform, test_transform = get_transform(
            test_transform,
            train_transform,
            val_transform,
            mean,
            std,
            augment,
            img_dim,
            padding_crop)

        trainset = UniformNoise(size=(img_ch, img_dim, img_dim))
        valset = UniformNoise(size=(img_ch, img_dim, img_dim))
        testset = UniformNoise(size=(img_ch, img_dim, img_dim))

    elif(dataset.lower() == 'g-noise'):
        img_dim = 32
        img_ch = 3
        num_classes = 1
        num_worker = 4
                
        if(mean == None or std == None):
            raise ValueError("Mean and std for textures no supported yet!")
                
        train_transform, val_transform, test_transform = get_transform(
            test_transform,
            train_transform,
            val_transform,
            mean,
            std,
            augment,
            img_dim,
            padding_crop)

        trainset = GaussianNoise(size=(img_ch, img_dim, img_dim))
        valset = GaussianNoise(size=(img_ch, img_dim, img_dim))
        testset = GaussianNoise(size=(img_ch, img_dim, img_dim))

    elif(dataset.lower() == 'isun'):
        if(mean == None or std == None):
            raise ValueError("Mean and std for textures no supported yet!")

        img_dim = 32
        img_ch = 3
        num_classes = 1
        num_worker = 4
        datapath ='Set Path'
                
        train_transform, val_transform, test_transform = get_transform(
            test_transform,
            train_transform,
            val_transform,
            mean,
            std,
            augment,
            img_dim,
            padding_crop)

        trainset = torchvision.datasets.ImageFolder(
            root=datapath,
            transform=train_transform)

        valset = torchvision.datasets.ImageFolder(
            root=datapath,
            transform=val_transform)

        testset = torchvision.datasets.ImageFolder(
            root=datapath,
            transform=test_transform)

    elif(dataset.lower() == 'imagenet'):
        if(mean == None):
            mean = [0.485, 0.456, 0.406]
        if(std == None):
            std = [0.229, 0.224, 0.225]

        img_dim = 224
        img_ch = 3
        num_classes = 1000
        num_worker = 40
        datapath = 'Set Path'
                
        train_transform, val_transform, test_transform = get_transform(
            test_transform,
            train_transform,
            val_transform,
            mean,
            std,
            augment,
            img_dim,
            padding_crop)

        trainset = torchvision.datasets.ImageFolder(
            root=datapath + 'train',
            transform=train_transform)

        valset = torchvision.datasets.ImageFolder(
            root=datapath + 'train',
            transform=val_transform)

        testset = torchvision.datasets.ImageFolder(root=datapath + 'val',
                                                   transform=test_transform)

    elif(dataset.lower() == 'imagenette'):
        if(mean == None):
            mean = [0.4625, 0.4580, 0.4295]
        if(std == None):
            std = [0.2813, 0.2774, 0.3006]

        img_dim = 64
        img_ch = 3
        num_classes = 10
        num_worker = 10
        datapath = 'Set Path'
                
        train_transform, val_transform, test_transform = get_transform(
            test_transform,
            train_transform,
            val_transform,
            mean,
            std,
            augment,
            img_dim,
            padding_crop,
            resize=True)

        trainset = torchvision.datasets.ImageFolder(
            root=datapath + 'train',
            transform=train_transform)

        valset = torchvision.datasets.ImageFolder(
            root=datapath + 'train',
            transform=val_transform)

        testset = torchvision.datasets.ImageFolder(
            root=datapath + 'val',
            transform=test_transform)

    elif dataset.lower() == 'coco_cap':
        if(mean == None):
            mean = [0, 0, 0]
        if(std == None):
            std = [1, 1, 1]

        img_dim = 256
        img_ch = 3
        num_classes = 1000
        num_worker = 8
        datapath ='Set Path'
        #datapath = 'Path for image net goes here' # Set path here
                
        train_transform, val_transform, test_transform = get_transform(
            test_transform,
            train_transform,
            val_transform,
            mean,
            std,
            augment,
            img_dim,
            padding_crop)        

        trainset = torchvision.datasets.CocoCaptions(
            root=os.path.join(datapath,'train2014'),
            annFile=os.path.join(datapath, 'annotations', 'captions_train2014.json'),
            transform=train_transform)

        valset = torchvision.datasets.CocoCaptions(
            root=os.path.join(datapath,'val2014'),
            annFile=os.path.join(datapath, 'annotations', 'captions_val2014.json'),
            transform=val_transform)

        testset = None

    elif (dataset.lower() == 'camelyon17'):
        if(mean == None):
            mean = [0, 0, 0]
        if(std == None):
            std = [1, 1, 1]

        img_dim = 96
        img_ch = 3
        num_classes = 2
        num_worker = 16

        dataset = get_dataset(
            dataset="camelyon17", 
            download=False, 
            root_dir=root_path)

        train_transform, val_transform, test_transform = get_transform(
            test_transform,
            train_transform,
            val_transform,
            mean,
            std,
            augment,
            img_dim,
            padding_crop)

        trainset = dataset.get_subset("train", transform=train_transform)
        valset = dataset.get_subset("train", transform=val_transform)
        testset =  dataset.get_subset("test", transform=test_transform)

    else:
        # Right way to handle exception in python 
        # see https://stackoverflow.com/questions/2052390/manually-raising-throwing-an-exception-in-python
        # Explains all the traps of using exception, does a good job!! I mean the link :)
        logger.error("Unsupported dataset")
        raise ValueError("Unsupported dataset")
    
    # Split the training dataset into training and validation sets
    logger.info('Forming the sampler for train and validation split')
    if index is None:
        num_train = len(trainset)
        ind = list(range(num_train))
    else:
        num_train = len(index)
        ind = index
    
    split = int(val_split * num_train)
    logger.info(f'Split counts 0 {split} {num_train}')

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(ind)

    train_idx, val_idx = ind[split:], ind[:split]
    valset = torch.utils.data.Subset(trainset, val_idx)
    trainset = torch.utils.data.Subset(trainset, train_idx)

    if args_num_workers is not None:
        num_worker = args_num_workers
    
    dataset_obj = create_dataloaders(
        dataset, 
        train_batch_size, 
        test_batch_size, 
        val_split, 
        augment, 
        padding_crop, 
        shuffle, 
        random_seed, 
        mean, 
        std, 
        train_transform, 
        test_transform, 
        val_transform, 
        logger, 
        img_dim, 
        img_ch, 
        num_classes, 
        num_worker, 
        trainset, 
        valset, 
        testset, 
        num_train,
        distributed)
    return dataset_obj

def create_dataloaders(
    dataset, 
    train_batch_size, 
    test_batch_size, 
    val_split, 
    augment, 
    padding_crop, 
    shuffle, 
    random_seed, 
    mean, 
    std, 
    train_transform, 
    test_transform, 
    val_transform, 
    logger, 
    img_dim, 
    img_ch, 
    num_classes, 
    num_worker, 
    trainset, 
    valset, 
    testset, 
    num_train,
    distributed):

    sampler = lambda dataset: DistributedSampler(dataset) if distributed else None
    train_shuffle = None if distributed else shuffle
    val_test_shuffle = None if distributed else False

    # Load dataloader
    logger.info('Loading data to the dataloader\n')
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=train_batch_size,
        shuffle=train_shuffle,
        pin_memory=False,
        sampler=sampler(trainset),
        num_workers=num_worker)

    val_loader =  torch.utils.data.DataLoader(
        valset,
        batch_size=train_batch_size,
        shuffle=val_test_shuffle,
        pin_memory=False,
        sampler=sampler(valset),
        num_workers=num_worker)

    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=test_batch_size,
        shuffle=val_test_shuffle,
        pin_memory=False,
        sampler=sampler(testset),
        num_workers=num_worker)

    transforms_dict = {
        'train': train_transform,
        'val': val_transform,
        'test': test_transform
    }

    return_dict = {
        'name': dataset,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'num_classes': num_classes,
        'train_length': num_train,
        'testset': testset,
        'mean' : mean,
        'std': std,
        'img_dim': img_dim,
        'img_ch': img_ch,
        'train_batch_size': train_batch_size,
        'test_batch_size': test_batch_size,
        'val_split': val_split,
        'padding_crop': padding_crop,
        'augment': augment,
        'random_seed': random_seed,
        'shuffle': shuffle,
        'transforms': Dict_To_Obj(**transforms_dict),
        'num_worker': num_worker
    }

    dataset_obj = Dict_To_Obj(**return_dict)
    return dataset_obj
