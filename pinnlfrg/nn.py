import torch
from torch import nn


class NN(nn.Module):
    """
    This class defines an NN architecture for gamma(l, phi).
    """

    def __init__(self, Nphi, m2, use_xavier=1, Nlayer=4, Nnode=256, std=1.e-3):
        super().__init__()

        self.Nphi = Nphi
        self.Nlayer = Nlayer
        self.Nnode = Nnode
        self.std = std
        self.m2 = m2
        self.m = m2**0.5
        self.use_xavier = use_xavier

        self.layer_omega = nn.Sequential()
        self.layer_omega_def()
        self._init_weights(self.layer_omega)

    def layer_omega_def(self):
        act = nn.Softplus
        # act = nn.Tanh
        # act = nn.SiLU

        # First hidden layer
        self.layer_omega.add_module(
            'omega start', nn.Linear(self.Nphi+1, self.Nnode))
        # Second to last hidden layers
        for i in range(self.Nlayer - 1):
            self.layer_omega.add_module(
                'omega activate {0}'.format(i), act())
            self.layer_omega.add_module(
                'omega Linear {0}'.format(i), nn.Linear(self.Nnode, self.Nnode))
        self.layer_omega.add_module('omega activate last', act())
        # Output layer
        self.layer_omega.add_module('omega last', nn.Linear(self.Nnode, 1))

    def _init_weights(self, module):
        for x in module:
            if isinstance(x, nn.Linear):
                if self.use_xavier == 1:
                    torch.nn.init.xavier_normal_(x.weight, gain=self.std)
                else:
                    x.weight.data.normal_(mean=0., std=self.std)
                if x.bias is not None:
                    x.bias.data.zero_()

    def forward(self, input):
        """
        input: batch of [l, p1, ..., pN]
        """
        # extracting l from batch of [l, p1, ... pN]
        ls = input[:, 0]
        lsv = ls.view(-1, 1)

        # extracting [p1, ... , pN] from batch of [l, p1, ... pN]
        ps = input[:, 1:]
        # rescaling p
        pstil = self.m * ps

        # calculation of gam(l,p) = NN(l,p) - NN(0,p)
        # NN(l,p) = N * omega(mp, l)
        lx_p = torch.cat((lsv, pstil), dim=1)
        l0x_p = torch.cat((0.*lsv, pstil), dim=1)
        gam = self.Nphi*(self.layer_omega(lx_p)-self.layer_omega(l0x_p))

        return gam
