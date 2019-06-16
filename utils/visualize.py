# visualize.py

import torch
import matplotlib  # .pyplot as plt
import torchvision
import numpy as np
from utils.misc import *

__all__ = ['make_image', 'show_batch', 'show_mask', 'show_mask_single']

# functions to show an image
def make_image(img, mean=(0, 0, 0), std=(1, 1, 1)):
    for i in range(0, 3):
        img[i] = img[i] * std[i] + mean[i]    # unnormalize
    npimg = img.numpy()
    return np.transpose(npimg, (1, 2, 0))

def gauss(x,a,b,c):
    return torch.exp(-torch.pow(torch.add(x,-b),2).div(2*c*c)).mul(a)

def colorize(x):
    ''' Converts a one-channel grayscale image to a color heatmap image '''
    if x.dim() == 2:
        torch.unsqueeze(x, 0, out=x)
    if x.dim() == 3:
        cl = torch.zeros([3, x.size(1), x.size(2)])
        cl[0] = gauss(x,.5,.6,.2) + gauss(x,1,.8,.3)
        cl[1] = gauss(x,1,.5,.3)
        cl[2] = gauss(x,1,.2,.3)
        cl[cl.gt(1)] = 1
    elif x.dim() == 4:
        cl = torch.zeros([x.size(0), 3, x.size(2), x.size(3)])
        cl[:,0,:,:] = gauss(x,.5,.6,.2) + gauss(x,1,.8,.3)
        cl[:,1,:,:] = gauss(x,1,.5,.3)
        cl[:,2,:,:] = gauss(x,1,.2,.3)
    return cl

def show_batch(images, Mean=(2, 2, 2), Std=(0.5,0.5,0.5)):
    images = make_image(torchvision.utils.make_grid(images), Mean, Std)
    matplotlib.pyplot.imshow(images)
    matplotlib.pyplot.show()


def show_mask_single(images, mask, Mean=(2, 2, 2), Std=(0.5,0.5,0.5)):
    im_size = images.size(2)

    # save for adding mask
    im_data = images.clone()
    for i in range(0, 3):
        im_data[:,i,:,:] = im_data[:,i,:,:] * Std[i] + Mean[i]    # unnormalize

    images = make_image(torchvision.utils.make_grid(images), Mean, Std)
    matplotlib.pyplot.subplot(2, 1, 1)
    matplotlib.pyplot.imshow(images)
    matplotlib.pyplot.axis('off')

    mask_size = mask.size(2)
    mask = (upsampling(mask, scale_factor=im_size/mask_size))
    mask = make_image(torchvision.utils.make_grid(0.3*im_data+0.7*mask.expand_as(im_data)))
    matplotlib.pyplot.subplot(2, 1, 2)
    matplotlib.pyplot.imshow(mask)
    matplotlib.pyplot.axis('off')

def show_mask(images, masklist, Mean=(2, 2, 2), Std=(0.5,0.5,0.5)):
    im_size = images.size(2)

    # save for adding mask
    im_data = images.clone()
    for i in range(0, 3):
        im_data[:,i,:,:] = im_data[:,i,:,:] * Std[i] + Mean[i]    # unnormalize

    images = make_image(torchvision.utils.make_grid(images), Mean, Std)
    matplotlib.pyplot.subplot(1+len(masklist), 1, 1)
    matplotlib.pyplot.imshow(images)
    matplotlib.pyplot.axis('off')

    for i in range(len(masklist)):
        mask = masklist[i].data.cpu()
        mask_size = mask.size(2)
        mask = (upsampling(mask, scale_factor=im_size/mask_size))

        mask = make_image(torchvision.utils.make_grid(0.3*im_data+0.7*mask.expand_as(im_data)))
        matplotlib.pyplot.subplot(1+len(masklist), 1, i+2)
        matplotlib.pyplot.imshow(mask)
        matplotlib.pyplot.axis('off')
