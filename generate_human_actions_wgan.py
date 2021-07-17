import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
from dataset import NTUSkeletonDataset
from torch.utils.data import Dataset, DataLoader
# import gan
from torch.autograd import Variable
import matplotlib.pyplot as plt
import time
import visualization as vs
import wandb

import logging
logging.propagate = False
logging.getLogger().setLevel(logging.ERROR)

# Root directory for dataset
dataroot = "processed_mutual_data_medium"

# Batch size during training
batch_size = 5

# Size of z latent vector (i.e. size of generator input)
latent_dim = 50

# Number of training epochs
num_epochs = 150

# Learning rate for optimizers
lr = 0.00005

clip_value = 0.01
n_critic = 5

trainset = NTUSkeletonDataset(root_dir=dataroot, pinpoint=21, pin_body=1)
trainloader = DataLoader(trainset, batch_size=batch_size,
                         shuffle=True, num_workers=4)
ngpu = 1


KEYPOINTS = 25
DIM = 2  # x and y
PERSON = 2
FRAME = 30
img_shape = (PERSON, KEYPOINTS, DIM)
warm_start = False
model_save_epoch = 15


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            # layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.ReLU())

            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=True),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, PERSON*KEYPOINTS*DIM),
        )

    def forward(self, z):
        img = self.model(z)
        img = img.reshape(img.shape[0], *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity


generator = Generator()
discriminator = Discriminator()
cuda = True if torch.cuda.is_available() else False
# cuda = False
if cuda:
    generator.cuda('cuda:0')
    discriminator.cuda('cuda:0')
    device = torch.device("cuda:0" if (
        torch.cuda.is_available() and ngpu > 0) else "cpu")
else:
    device = torch.device('cpu')


optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=lr)
optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=lr)
# optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr)
# optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr)
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
batches_done = 0


def valid_data(data):
    result = np.array([]).reshape((0, 2, 25, 2))
    ori_data = data
    invalid_counter = 0
    for i in range(data.shape[0]):
        count_x_0 = 0
        count_y_0 = 0
        count_x_1 = 0
        count_y_1 = 0

        for p in range(data.shape[2]):
            if(data[i, 0, p, 0] >= 0.2):
                count_x_0 += 1
            if(data[i, 0, p, 1] != 0):
                count_y_0 += 1
            if(data[i, 1, p, 0] >= 0.2):
                count_x_1 += 1
            if(data[i, 1, p, 1] != 0):
                count_y_1 += 1
        if((count_x_0 >= 5 and count_y_0 >= 20) and (count_x_1 >= 5 and count_y_1 >= 20)):
            good_data = data[i, :, :, :]
            good_data = np.expand_dims(good_data, axis=0)

            result = np.concatenate((result, good_data))
        else:
            invalid_counter += 1
    while(result.shape[0] != ori_data.shape[0]):
        rand_num = np.random.randint(low=0, high=result.shape[0])
        add_batch = result[rand_num, :, :, :]
        add_batch = np.expand_dims(add_batch, axis=0)
        result = np.concatenate((result, add_batch))
    return result


wandb.init(project="1699project1")

# Magic
wandb.watch((generator, discriminator), log='all')

total_invalid_frame = 0
for epoch in range(num_epochs):
    epoch_start = time.time()
    valid_error = False
    for i, data in enumerate(trainloader):
        valid_error = False
        frame = data.shape[3]
        PERSON = data.shape[1]
        # f_num = np.random.randint(low=0, high=100)
        # f_num = np.random.randint(low=30, high=100)
        data = data.transpose(1, 2)
        s_num = (np.random.randint(low=0, high=3))*30
        data = data[:, s_num:s_num+30, :, :]
        data = data.reshape((data.shape[0]*data.shape[1]),
                            data.shape[2], data.shape[3], data.shape[4])
        p_num = 0

        try:
            input_x = valid_data(data)
        except:
            valid_error = True
            total_invalid_frame += 1
            continue
        input_x = torch.from_numpy(input_x)
        real_skeleton = Variable(input_x.type(Tensor)).to(device)

        optimizer_D.zero_grad()
        z = Variable(Tensor(np.random.normal(
            0, 1, (data.shape[0], latent_dim)))).to(device)

        # Generate a batch of skeleton
        fake_skeleton = generator(z).detach()

        # Adversarial loss
        loss_D = -torch.mean(discriminator(real_skeleton)) + \
            torch.mean(discriminator(fake_skeleton))

        loss_D.backward()
        optimizer_D.step()

        # clip weights of discriminator
        for p in discriminator.parameters():
            p.data.clamp_(-clip_value, clip_value)

        # Train the generator every n_critic iterations
        if i % n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Generate a batch of images
            gen_skeleton = generator(z)

            # Adversial loss
            loss_G = -torch.mean(discriminator(gen_skeleton))

            loss_G.backward()
            optimizer_G.step()

            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [invalid_batch: %d]"
                  % (epoch, num_epochs, batches_done % len(trainloader), len(trainloader), loss_D.item(), loss_G.item(), total_invalid_frame)
                  )
        batches_done += 1
        wandb.log({
            "epoch": "%d_%d" % (epoch, batches_done),
            "d loss": loss_D.item(),
            "g loss": loss_G.item()
        })

    epoch_end = time.time()
    print("time eplased: ", epoch_end-epoch_start)
    model_saving_dir = 'simple_wgan_exp/model_checkpoint/cp_epoch_%d_.tar' % (
        epoch)
    if (epoch % model_save_epoch == 0):
        torch.save({
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            'optimizer_D_state_dict': optimizer_D.state_dict(),
            'epoch': epoch,
            'loss_D': loss_D.item(),
            'loss_G': loss_G.item(),
            'total_invalid_frame': total_invalid_frame
        }, model_saving_dir)
    if valid_error is False:
        gen_skeleton_np = gen_skeleton.transpose(0, 1)
        gen_skeleton_np = gen_skeleton_np.cpu().detach().numpy()
        # gen_skeleton_np = np.expand_dims(gen_skeleton_np, axis=0)
        firstSkeleton = gen_skeleton_np[0:1, :, :, 0:2]
        secondSkeleton = gen_skeleton_np[1:2, :, :, 0:2]

        fig, ax = vs.draw_skeleton(firstSkeleton, step=2)

        plt.savefig('simple_wgan_exp/fig/fig_2p_0/epoch%d_%3f_%3f_p%d.jpg' %
                    (epoch, loss_D.item(), loss_G.item(), 0))
        fig, ax = vs.draw_skeleton(secondSkeleton, step=2)
        plt.savefig('simple_wgan_exp/fig/fig_2p_1/epoch%d_%3f_%3f_p%d.jpg' %
                    (epoch, loss_D.item(), loss_G.item(), 1))
        fig, ax = vs.draw_skeleton(
            gen_skeleton_np[:, :, :, 0:2], step=2, max_bodies=2)
        plt.savefig('simple_wgan_exp/fig/fig_2p_2/epoch%d_%3f_%3f_p%d.jpg' %
                    (epoch, loss_D.item(), loss_G.item(), 2))
    else:
        valid_error = False
print("total invalid frame: ", total_invalid_frame)
