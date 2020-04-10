import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
from dataset import NTUSkeletonDataset
from torch.utils.data import Dataset, DataLoader
import GAN
from torch.autograd import Variable
import matplotlib.pyplot as plt
import time


# Root directory for dataset
dataroot = "data/small_dataset"

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 5

# Number of channels in the training images. For color images this is 3
nc = 2

# Size of z latent vector (i.e. size of generator input)
latent_dim = 20

# Number of training epochs
num_epochs = 100

# Learning rate for optimizers
lr = 0.0001

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

cuda = torch.cuda.is_available()
clip_value = 0.01
n_critic = 5


trainset = NTUSkeletonDataset(root_dir=dataroot, pinpoint=10)
trainloader = DataLoader(trainset, batch_size=batch_size,
                         shuffle=True, num_workers=4)

generator = GAN.Gen0(latent_dim)
discriminator = GAN.Dis0()

if cuda:
    generator.cuda('cuda:0')
    discriminator.cuda('cuda:0')
    device = torch.device("cuda:0" if (
        torch.cuda.is_available() and ngpu > 0) else "cpu")


optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

epoch_loss = np.zeros((num_epochs, 2, len(trainloader)//n_critic+1))

for epoch in range(num_epochs):
    j = 0
    epoch_start = time.time()
    for i, data in enumerate(trainloader):
        size = (*data.size()[:3], 50)
        data = data.reshape(size)
        real_skeleton = Variable(data.type(Tensor)).to(device)

        optimizer_D.zero_grad()

        # sample noise as generator input
        z = torch.randn(real_skeleton.size(0), latent_dim).to(device)

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

            epoch_loss[epoch, 0, j] = loss_G.item()
            epoch_loss[epoch, 1, j] = loss_D.item()
            j += 1

    epoch_end = time.time()
    print('[%d] time eplased: ' % epoch, epoch_end-epoch_start)

fig, ax = plt.subplots()
fig.tight_layout()
t = np.arange(num_epochs)
mean_G = epoch_loss[:,0,:].mean(axis=1)
mean_D = epoch_loss[:,1,:].mean(axis=1)
std_G = epoch_loss[:,0,:].std(axis=1)
std_D = epoch_loss[:,1,:].std(axis=1)
ax.plot(t, mean_G, label='G')
ax.plot(t, mean_D, label='D')
ax.fill_between(t, mean_G + std_G, mean_G - std_G, alpha=0.1)
ax.fill_between(t, mean_D + std_D, mean_D - std_D, alpha=0.1)
fig.legend()
fig.savefig('train.png')
