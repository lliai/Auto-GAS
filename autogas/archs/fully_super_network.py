# Code borrowed from the "EAGAN" by marsggbo
# Original code: https://github.com/marsggbo/EAGAN

import torch
from torch import nn

from gaswot.archs.fully_basic_network import Cell, DisCell, OptimizedDisBlock


class Generator(nn.Module):

    def __init__(self, args, genotypes):
        super(Generator, self).__init__()
        self.args = args
        self.ch = args.gf_dim
        self.bottom_width = args.bottom_width
        self.base_latent_dim = args.latent_dim // 3
        self.l1 = nn.Linear(self.base_latent_dim,
                            (self.bottom_width**2) * args.gf_dim)
        self.l2 = nn.Linear(self.base_latent_dim,
                            ((self.bottom_width * 2)**2) * args.gf_dim)
        self.l3 = nn.Linear(self.base_latent_dim,
                            ((self.bottom_width * 4)**2) * args.gf_dim)
        self.cell1 = Cell(
            args.gf_dim, args.gf_dim, 'nearest', genotypes[0], num_skip_in=0)
        self.cell2 = Cell(
            args.gf_dim, args.gf_dim, 'bilinear', genotypes[1], num_skip_in=1)
        self.cell3 = Cell(
            args.gf_dim, args.gf_dim, 'nearest', genotypes[2], num_skip_in=2)
        self.to_rgb = nn.Sequential(
            nn.BatchNorm2d(args.gf_dim), nn.ReLU(),
            nn.Conv2d(args.gf_dim, 3, 3, 1, 1), nn.Tanh())

    def forward(self, z):
        h = self.l1(z[:, :self.base_latent_dim])\
            .view(-1, self.ch, self.bottom_width, self.bottom_width)

        n1 = self.l2(z[:, self.base_latent_dim:self.base_latent_dim * 2])\
            .view(-1, self.ch, self.bottom_width * 2, self.bottom_width * 2)

        n2 = self.l3(z[:, self.base_latent_dim * 2:])\
            .view(-1, self.ch, self.bottom_width * 4, self.bottom_width * 4)

        h1_skip_out, h1 = self.cell1(h)
        h2_skip_out, h2 = self.cell2(h1 + n1, (h1_skip_out, ))
        _, h3 = self.cell3(h2 + n2, (h1_skip_out, h2_skip_out))
        output = self.to_rgb(h3)

        return output


# AutoGAN-D
class Discriminator(nn.Module):

    def __init__(self, args, genotypes, activation=nn.ReLU()):
        super(Discriminator, self).__init__()
        # args.df_dim is the number of filters in the first layer of the discriminator
        self.ch = args.df_dim
        self.activation = activation
        # The first layer of the discriminator is an optimized convolutional layer
        self.block1 = OptimizedDisBlock(args, 3, self.ch)
        # The next three layers of the discriminator are the same cell
        # The number of filters in the first layer of the cell is self.ch
        # The number of filters in the second layer of the cell is self.ch
        self.block2 = DisCell(
            args,
            self.ch,
            self.ch,
            activation=activation,
            genotype=genotypes[0])
        self.block3 = DisCell(
            args,
            self.ch,
            self.ch,
            activation=activation,
            genotype=genotypes[1])
        self.block4 = DisCell(
            args,
            self.ch,
            self.ch,
            activation=activation,
            genotype=genotypes[2])
        # The last layer of the discriminator is a linear layer


class simple_Discriminator(nn.Module):

    def __init__(self):
        super(simple_Discriminator, self).__init__()
        # first layer
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
        )
        # second layer
        self.block2 = nn.Sequential(
            nn.Conv2d(128, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 4, stride=2, padding=1),
        )
        # third layer
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        # output layer
        self.l4 = nn.Linear(128, 1, bias=False)
        self.l4 = nn.utils.spectral_norm(self.l4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        layers = [self.block1, self.block2, self.block3]
        model = nn.Sequential(*layers)
