# main.py

import os
import sys
from dataloader import *
import torch.utils.data
from train import MaxentNet

sys.path.append('../')


# os.environ['CUDA_VISIBLE_DEVICES'] = "6"
use_cuda = torch.cuda.is_available()
device = torch.device("cuda") if use_cuda else "cpu"

## Set the embedding dimension
embed_length = 128

print('==> Loading and Preparing data..')
## Uncomment these for other datasets
# data = AdultDataLoader()
# data = ExtendedYaleBDataLoader()
# data = GermanDataLoader()
# data = CIFAR10DataLoader(embed_length=embed_length)
# data = GaussianDataLoader()

data = CIFAR100DataLoader(embed_length=embed_length)
print('==> Building models..')
data.load()  # <--This method loads all the parameters for the data including neural-net models and optimizer

## Initialize the data loaders
trainloader = torch.utils.data.DataLoader(data.trainset,
                                          batch_size=data.train_batch_size,
                                          shuffle=True,
                                          num_workers=40
                                          )
testloader = torch.utils.data.DataLoader(data.testset,
                                         batch_size=data.test_batch_size,
                                         shuffle=False,
                                         num_workers=40)

runs = np.arange(5, 6)

## this is for MaxEnt-ARL
alphalist = [0.1, 0.2, 0.3, 0.32, 0.34, 0.36, 0.38, 0.4, 0.5, 0.6]

## this is for ML-ARL
# alphalist = [0.1, 0.2, 0.3, 0.4, 0.42, 0.44, 0.46, 0.48, 0.5, 0.6]

# Run this multiple times
for r in runs:
    for alpha in alphalist:
        name = 'cifar100_' + str(float(alpha))[:4] + '_' + str(r)
        target_name = name + "_target_" + ".ckpt"
        exist = os.path.isfile('checkpoint/' + target_name)

        if exist:
            continue
        trainer = MaxentNet(data,
                            train_loader=trainloader,
                            test_loader=testloader,
                            total_epoch=150,
                            alpha=alpha,
                            use_cuda=use_cuda,
                            ckpt_filename=name,
                            privacy_flag=False,
                            privacy_option='maxent-arl', # 'ml-arl' for ML-ARL
                            resume=False,
                            resume_filename=name + '.ckpt',
                            print_interval_train=10,
                            print_interval_test=10
                            )

        trainer.train()
        trainer.train_adversary(model_filename=name+'.ckpt', total_epoch=150)
        trainer.train_target(model_filename=name + '.ckpt', total_epoch=100)