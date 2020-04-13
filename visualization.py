import numpy as np
import matplotlib.pyplot as plt
from itertools import product

bone_list = ((1, 2), (2, 21), (21, 3), (3, 4),
             (21, 5), (5, 6), (6, 7), (7, 8), (8, 22), (8, 23),
             (21, 9), (9, 10), (10, 11), (11, 12), (12, 24), (12, 25),
             (1, 13), (13, 14), (14, 15), (15, 16),
             (1, 17), (17, 18), (18, 19), (19, 20))
bone_list = np.array(bone_list) - 1

def train_plot(fname, lnames):
    epoch_loss = np.load(fname)

    fig, ax = plt.subplots()
    fig.tight_layout()
    t = np.arange(epoch_loss.shape[0])
    for k, l in enumerate(lnames):
        mean = epoch_loss[:,k,:].mean(axis=1)
        std = epoch_loss[:,k,:].std(axis=1)
        ax.plot(t, mean, label=l)
        ax.fill_between(t, mean + std, mean - std, alpha=0.1)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    fig.legend()
    fig.savefig(fname[:-4] + '.png')

def draw_skeleton(skeleton, step=3, max_bodies=1, max_frames=(4, 4),
                  color='r', s=10):
    """
    skeleton: numpy.ndarray
        Must have dim (# bodies, #frames, # keypoints, xy)
    max_frames: tuple
        Rows and cols of frames
    """
    fig, ax = plt.subplots(*max_frames, sharex='all', sharey='all')
    fig.set_size_inches(w=3*max_frames[0], h=3*max_frames[1])
    fig.tight_layout(pad=0.1)
    for p, q in product(range(max_frames[0]), range(max_frames[1])):
        i = step * (p * max_frames[0] + q)
        for j in range(max_bodies):
            ax[p, q].scatter(skeleton[j, i, :, 0], skeleton[j, i, :, 1], s)
            ax[p, q].text(0, 0, i, transform=ax[p, q].transAxes)
            for bone in bone_list:
                x = [skeleton[j, i, bone[0], 0], skeleton[j, i, bone[1], 0]]
                y = [skeleton[j, i, bone[0], 1], skeleton[j, i, bone[1], 1]]
                ax[p, q].plot(x, y, color)

    return fig, ax

if __name__ == '__main__':
    vae0fname = 'vae0_epoch_loss.npy'
    vae0labels = ('Rec', 'KL', 'GAN', 'critic real', 'critic fake')
    train_plot(vae0fname, vae0labels)

    gan0fname = 'gan0_epoch_loss.npy'
    gan0labels = ('G', 'critic real', 'critic fake')
    train_plot(gan0fname, gan0labels)
