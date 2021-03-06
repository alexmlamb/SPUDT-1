from torch import nn


class Encoder(nn.Module):
    def __init__(self, i_dim, h_dim, n_classes, **kwargs):
        super(Encoder, self).__init__()
        self.x = nn.Sequential(
            nn.Conv2d(i_dim, h_dim, 1, 1, 1),
            nn.BatchNorm2d(h_dim, eps=1e-8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(h_dim, h_dim, 3, 1, 1),
            nn.BatchNorm2d(h_dim, eps=1e-8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(h_dim, h_dim, 3, 1, 1),
            nn.BatchNorm2d(h_dim, eps=1e-8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(h_dim, h_dim, 3, 1, 1),
            nn.BatchNorm2d(h_dim, eps=1e-8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(h_dim, h_dim, 3, 1, 1),
            nn.BatchNorm2d(h_dim, eps=1e-8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(h_dim, n_classes, 1, 1, 0),
            nn.AdaptiveAvgPool2d((1, 1)))

    def forward(self, x):
        return self.x(x).squeeze()


class Contrastive(nn.Module):
    def __init__(self, h_dim, n_classes, **kwargs):
        super(Contrastive, self).__init__()
        h_dim = h_dim
        self.x = nn.Sequential(nn.Linear(n_classes, h_dim),
                               nn.LeakyReLU(0.2, inplace=True),
                               nn.Linear(h_dim, h_dim),
                               nn.LeakyReLU(0.2, inplace=True),
                               nn.Linear(h_dim, h_dim),
                               nn.LeakyReLU(0.2, inplace=True),
                               nn.Linear(h_dim, h_dim),
                               nn.LeakyReLU(0.2, inplace=True),
                               nn.Linear(h_dim, 1))

    def forward(self, a):
        return self.x(a).squeeze()
