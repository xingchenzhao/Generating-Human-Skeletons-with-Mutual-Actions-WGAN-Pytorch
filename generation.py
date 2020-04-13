import torch
from visualization import draw_skeleton

def load(fname, model, config=None, state_dict=True):
    cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda else "cpu")
    m = torch.load(fname, map_location=device)
    if state_dict:
        m = model(*config)
        m.load_state_dict()
    return m

def gen_single_frames(model, in_dim, max_frames=(4, 4), fname='samples_sf'):
    z = torch.randn(max_frames[0] * max_frames[1], in_dim)
    x = model(z=z)
    x = x.reshape((1, -1, 25, 2)).detach().numpy()
    fig, ax = draw_skeleton(x, step=1)
    fig.savefig(fname + '.png')

if __name__ == '__main__':
    import GAN
    import sys
    dims = list(map(int, sys.argv[3:]))
    if sys.argv[1] == 'vae':
        m0 = GAN.VAE0
        config = (dims[0], dims[1:])
    elif sys.argv[1] == 'gen0':
        m0 = GAN.Gen0
        config = (dims[0],)

    m = load(sys.argv[2], m0, config)
    gen_single_frames(m, config[0], fname='samples_' + sys.argv[2])
