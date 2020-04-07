import torch
from torch import nn, optim

KEYPOINTS = 25
DIM = 2 # x and y

def linear_block(in_dim, out_dim, batch_norm=True):
    l = [nn.Linear(in_dim, out_dim), nn.ReLU()]
    if batch_norm:
        l.insert(1, nn.BatchNorm1d(out_dim))
    return l


def output_block(in_dim, out_dim):
    return [nn.Dropout(0.5), nn.Linear(in_dim, out_dim)]


class Basic(nn.Module):
    def __init__(self, in_dim, out_dim, batch_norm=True):
        super(Basic, self).__init__()

        self.block1 = nn.Sequential(
            *linear_block(in_dim, 4096, batch_norm)
        )

        l = []
        for _ in range(2):
            l += linear_block(4096, 4096, batch_norm)
        self.block2 = nn.Sequential(*l)

        self.block3 = nn.Sequential(
            *output_block(4096, out_dim)
        )

    def forward(self, z):
        b1 = self.block1(z)
        b2 = self.block2(b1)
        return self.block3(b1 + b2)


class Gen0(Basic):
    def __init__(self, z0_dim):
        """
        z_dim: int
            The dimension of the input noise z
        """
        self.z0_dim = z0_dim
        super(Gen0, self).__init__(z0_dim, KEYPOINTS * DIM)

    def forward(self, seq, separate_xy=False):
        l = seq.size()
        if len(l) == 3: # merge 0 and 1 axis
            t = seq.size(1)
            o = super(Gen0, self).forward(seq.reshape((-1, self.z0_dim)))

        else: # Assume it is 2 or 1
            o = super(Gen0, self).forward(seq)

        if separate_xy:
            return o.reshape((*l[:-1], KEYPOINTS, DIM))
        else:
            return o.reshape((*l[:-1], KEYPOINTS * DIM))

class Dis0(Basic):
    def __init__(self):
        super(Dis0, self).__init__(KEYPOINTS * DIM, 1, False)


class GenPs(Basic):
    def __init__(self, t, z0_dim, z_dim):
        """
        t: int
            The length of the target sequence
        z0_dim: int
            The dimension of the input noise z0
        z_dim: int
            The dimension of the input noise z
        """
        self.t = t
        self.z0_dim = z0_dim

        super(GenPs, self).__init__(z0_dim + z_dim, z0_dim * (t - 1))

    def forward(self, z0, z):
        z_ = torch.cat((z0, z), dim=-1)
        o0 = super(GenPs, self).forward(z_)
        o1 = torch.cat((z0, o0), dim=-1)
        return o1.reshape(z0.size(0), self.t, self.z0_dim)


class DisPs(nn.Module):
    def __init__(self, t, hidden_dim):
        """
        t: int
            The length of the target sequence
        hidden_dim: int
            The dimension of the hidden states of the LSTM
        """
        super(DisPs, self).__init__()

        self.t = t
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(KEYPOINTS * DIM, hidden_dim,
                            batch_first=True, bidirectional=True)
        self.l = Basic(t * 2 * hidden_dim, 1, False)

    def forward(self, seq):
        bs = seq.size(0)
        h0 = torch.randn(2, bs, self.hidden_dim)
        c0 = torch.randn(2, bs, self.hidden_dim)

        z, _ = self.lstm(seq, (h0, c0))

        return self.l(z.reshape((bs, -1)))


# Testing
if __name__ == '__main__':
    z0_dim = 10
    z_dim = 20
    hidden_dim = 30
    t = 40
    bs = 50

    g0 = Gen0(z0_dim)
    d0 = Dis0()
    gps = GenPs(t, z0_dim, z_dim)
    dps = DisPs(t, hidden_dim)

    z0 = torch.randn(bs, z0_dim)
    z = torch.randn(bs, z_dim)

    s0 = g0(z0)
    print('Gen0', s0.size())
    print('Dis0', d0(s0).size())

    s = gps(z0, z)
    o = g0(s)
    print('GenPs', s.size())
    print('DisPs', dps(o).size())

    o = g0(s, separate_xy=True)
    print('Final output', o.size())
    assert not torch.isnan(o).any() # Should not contain NaN
