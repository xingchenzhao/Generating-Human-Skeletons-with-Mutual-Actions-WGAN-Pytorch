
from __future__ import print_function
# %matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from dataset import NTUSkeletonDataset
from torch.utils.data import Dataset, DataLoader
import gan
from torch.autograd import Variable
# %matplotlib inline
import matplotlib.pyplot as plt
import time


# Root directory for dataset
dataroot = "processed_mutual_data"

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 512

# Number of channels in the training images. For color images this is 3
nc = 2

# Size of z latent vector (i.e. size of generator input)
latent_dim = 20

# Number of training epochs
num_epochs = 100

# Learning rate for optimizers
lr = 0.00005

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

cuda = True if torch.cuda.is_available() else False
clip_value = 0.01
n_critic = 5
sample_interval = 400


trainset = NTUSkeletonDataset(root_dir=dataroot, pinpoint=10)
trainloader = DataLoader(trainset, batch_size=batch_size,
                         shuffle=True, num_workers=4)

generator = gan.Gen0(latent_dim)
discriminator = gan.Dis0()

if cuda:
    generator.cuda('cuda:0')
    discriminator.cuda('cuda:0')
    device = torch.device("cuda:0" if (
        torch.cuda.is_available() and ngpu > 0) else "cpu")


optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=lr)
optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=lr)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

batches_done = 0

for epoch in range(num_epochs):
    epoch_start = time.time()
    for i, data in enumerate(trainloader):
        # print(data.shape)
        frame = data.shape[3]
        person = data.shape[1]
        input_x = None
        # p_num = np.random.randint(low=0, high=2)
        f_num = np.random.randint(low=30, high=100)
        # f_num = 50
        p_num = 0
        # for f_num in range(frame):
        #     for p_num in range(person):
        input_x = data[:, p_num, :, f_num, 0:2]
        #    print(input_x.shape)
        real_skeleton = Variable(input_x.type(Tensor)).to(device)

        optimizer_D.zero_grad()

        # sample noise as generator input
        z = torch.randn(batch_size, latent_dim).to(device)

        # Generate a batch of fake skeleton
        fake_skeleton = generator(z).detach()
        # adversarial loss
        loss_D = -torch.mean(discriminator(real_skeleton)) + \
            torch.mean(discriminator(fake_skeleton))
        loss_D.backward()
        optimizer_D.step()

        # clip weights of discriminator
        for p in discriminator.parameters():
            p.data.clamp_(-clip_value, clip_value)
         # Train the generator every n_critic iterations:
        if i % n_critic == 0:
            optimizer_G.zero_grad()
            # Generate a batch of
            gen_skeleton = generator(z)
            # adversarial loss
            loss_G = -torch.mean(discriminator(gen_skeleton))

            loss_G.backward()
            optimizer_G.step()
            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [f num: %d / p num: %d]"
                  % (epoch, num_epochs, batches_done % len(trainloader), len(trainloader), loss_D.item(), loss_G.item(), f_num, p_num)
                  )
            # if batches_done % sample_interval == 0:
        batches_done += 1
    fig = plt.figure()
    plt.axis('off')
    # result = gen_skeleton.cpu().numpy()

    f, axes = plt.subplots(
        nrows=3, ncols=3, sharex=True, sharey=True)

    for i in range(3):
        for j in range(3):
            index = np.random.randint(low=0, high=128)
            axes[i][j].scatter(
                gen_skeleton.cpu().data[index, :, 0], gen_skeleton.cpu().data[index, :, 1])
            axes[i][j].set_xlim(-1, 1)
            axes[i][j].set_ylim(-1, 1)
            axes[i][j].set_xlabel(index)
    # for i in range(10):
    #     for j in range(10):
    #         index = i*10+j
    #         axes[i][j].scatter(
    #             gen_skeleton.cpu().data[index, :, 0], gen_skeleton.cpu().data[index, :, 1])
    rand_num = np.random.randint(low=0, high=4096)
    plt.savefig('fig/epoch%d.jpg' %
                (epoch))
    epoch_end = time.time()
    print("time eplased: ", epoch_end-epoch_start)
