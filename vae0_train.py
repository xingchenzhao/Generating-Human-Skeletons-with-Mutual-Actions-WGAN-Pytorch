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

# Batch size during training
batch_size = 5

# Size of z latent vector (i.e. size of generator input)
z0_dim = 10
hidden_dim = 30

# Number of training epochs
num_epochs = 100

# Learning rate for optimizers
lr = 0.0001

clip_value = 0.01
n_critic = 5

trainset = NTUSkeletonDataset(root_dir=dataroot, pinpoint=10, merge=2)
trainloader = DataLoader(trainset, batch_size=batch_size,
                         shuffle=True, num_workers=4)

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

generator = GAN.VAE0(z0_dim, hidden_dim).to(device)
discriminator = GAN.Dis0().to(device)

optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr)

epoch_loss = np.zeros((num_epochs, 4, len(trainloader)//n_critic+1))

for epoch in range(num_epochs):
    j = 0
    epoch_start = time.time()
    for i, data in enumerate(trainloader):
        size = (-1, data.size(-1))
        data = data.reshape(size)

        real_skeleton = Variable(data.type(Tensor)).to(device)

        optimizer_D.zero_grad()

        # Generate a batch of fake skeleton
        fake_rec, fake_stn, _ = generator(real_skeleton)
        fake_rec = fake_rec.detach()
        fake_stn = fake_stn.detach()

        # losses
        gan_loss = -torch.mean(discriminator(real_skeleton)) \
                   + torch.mean(discriminator(fake_rec)) \
                   + torch.mean(discriminator(fake_stn))

        loss_D = gan_loss
        loss_D.backward()
        optimizer_D.step()

        # clip weights of discriminator
        for p in discriminator.parameters():
            p.data.clamp_(-clip_value, clip_value)

        # Train the generator every n_critic iterations:
        if i % n_critic == 0:
            optimizer_G.zero_grad()

            fake_rec, fake_stn, (mu, logvar, std) = generator(real_skeleton)
            rec_loss = torch.mean((fake_rec - real_skeleton)**2)
            kl_loss = -0.5 * torch.mean(logvar - std**2 - mu**2 + 1)
            gan_loss = torch.mean(discriminator(fake_rec)) \
                      + torch.mean(discriminator(fake_stn))

            loss_G = rec_loss + kl_loss - gan_loss

            loss_G.backward()
            optimizer_G.step()

            epoch_loss[epoch, 0, j] = rec_loss.item()
            epoch_loss[epoch, 1, j] = kl_loss.item()
            epoch_loss[epoch, 2, j] = loss_D.item()
            epoch_loss[epoch, 3, j] = loss_G.item()
            j += 1

    epoch_end = time.time()
    print('[%d] time eplased: %.3f' % (epoch, epoch_end-epoch_start))
    print('\tRec', epoch_loss[epoch, 0].mean(axis=-1))
    print('\tKL', epoch_loss[epoch, 1].mean(axis=-1))
    print('\tLoss D', epoch_loss[epoch, 2].mean(axis=-1))
    print('\tLoss G', epoch_loss[epoch, 3].mean(axis=-1))

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
