import os
import copy
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
z0_dim = 20
hidden_dim = [50, 30]

# Number of training epochs
num_epochs = 200

# Learning rate for optimizers
lrG = 0.00005
lrD = 0.00005

clip_value = 0.01
n_critic = 20

trainset = NTUSkeletonDataset(root_dir=dataroot, pinpoint=1, merge=2)
trainloader = DataLoader(trainset, batch_size=batch_size,
                         shuffle=True, num_workers=4)

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

generator = GAN.VAE0(z0_dim, hidden_dim).to(device)
discriminator = GAN.Dis0().to(device)

optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=lrG)
optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=lrD)

epoch_loss = np.zeros((num_epochs, 5, len(trainloader)//n_critic+1))

mse = nn.MSELoss()

for epoch in range(num_epochs):
    j = 0
    epoch_start = time.time()
    for i, data in enumerate(trainloader):
        size = (-1, data.size(-1))
        data = data.reshape(size)

        optimizer_D.zero_grad()

        real_skeleton = Variable(data.type(Tensor)).to(device)

        critic_real = -torch.mean(discriminator(real_skeleton))
        # critic_real.backward()

        # Generate a batch of fake skeleton
        fake_rec, fake_stn, _ = generator(real_skeleton)

        fake_rec = fake_rec.detach()
        fake_stn = fake_stn.detach()

        # losses
        critic_fake = torch.mean(discriminator(fake_rec)) \
                      + torch.mean(discriminator(fake_stn))
        # critic_fake.backward()

        loss_D = critic_real + critic_fake
        loss_D.backward()

        optimizer_D.step()

        # clip weights of discriminator
        for p in discriminator.parameters():
            p.data.clamp_(-clip_value, clip_value)

        # Train the generator every n_critic iterations:
        if i % n_critic == n_critic - 1:
            optimizer_G.zero_grad()

            fake_rec, fake_stn, (mu, logvar, std) = generator(real_skeleton)

            rec_loss = mse(fake_rec, real_skeleton)
            # rec_loss.backward()

            kl_loss = -0.5 * torch.sum(logvar - std**2 - mu**2 + 1)
            # kl_loss.backward()

            gan_loss = -torch.mean(discriminator(fake_rec)) \
                       -10 * torch.mean(discriminator(fake_stn))
            # gan_loss.backward()

            loss_G = rec_loss + kl_loss + gan_loss
            loss_G.backward()

            optimizer_G.step()

            for k, l in enumerate((rec_loss, kl_loss, gan_loss,
                                   critic_real, critic_fake)):
                epoch_loss[epoch, k, j] = l.item()
            j += 1


    epoch_end = time.time()
    print('[%d] time eplased: %.3f' % (epoch, epoch_end-epoch_start))
    for k, l in enumerate(('Rec', 'KL', 'G', 'critic real', 'critic fake')):
        print('\t', l, epoch_loss[epoch, k].mean(axis=-1))

    if epoch % 20 == 19:
        m = copy.deepcopy(generator.state_dict())
        torch.save(m, 'vae0_%d.pt' % epoch)

np.save('vae0_epoch_loss.npy', epoch_loss)
