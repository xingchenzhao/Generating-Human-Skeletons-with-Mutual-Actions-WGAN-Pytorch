import numpy as np
import matplotlib.pyplot as plt

def train_plot(fname):
    epoch_loss = np.load(fname)

    fig, ax = plt.subplots()
    fig.tight_layout()
    t = np.arange(num_epochs)
    for k, l in enumerate():
        mean = epoch_loss[:,k,:].mean(axis=1)
        std = epoch_loss[:,k,:].std(axis=1)
        ax.plot(t, mean, label=l)
        ax.fill_between(t, mean + std, mean - std, alpha=0.1)
    fig.legend()
    fig.savefig(fname[:-4] + '.png')


if __name__ == '__main__':
    vae0fname = 'vae0_epoch_loss.npy'
    vae0labels = ('Rec', 'KL', 'GAN', 'critic real', 'critic fake')

    gan0fname = 'gan0_epoch_loss.npy'
    gan0labels = ('G', 'critic real', 'critic fake')
