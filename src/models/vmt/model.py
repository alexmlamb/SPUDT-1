import torch
from torch import nn

from models.vmt.attentive_densenet import AttentiveDensenet

#from attentive_densenet import AttentiveDensenet

class Discriminator(nn.Module):
    def __init__(self, i_dim, kernel_dim, h_dim, nc, **kwargs):
        super(Discriminator, self).__init__()
        assert kernel_dim % 16 == 0, "kernel_dim has to be a multiple of 16"

        self.x = nn.Sequential(nn.Linear(h_dim*8*8+nc, 100),
                               nn.ReLU(inplace=True),
                               nn.Linear(100, 100),
                               nn.ReLU(inplace=True),
                               nn.Linear(100, 100),
                               nn.ReLU(inplace=True),
                               nn.Linear(100, 100),
                               nn.ReLU(inplace=True),
                               nn.Linear(100, 1))
        self.init()

    def init(self):
        for layer in self.x:
            if 'Linear' in layer.__class__.__name__:
                nn.init.kaiming_normal_(layer.weight, mode='FAN_IN', nonlinearity='relu')
                layer.bias.data.fill_(0)

    def forward(self, x, c):
        o = torch.cat((x.view(x.shape[0], -1), c.squeeze()), 1)
        return self.x(o).squeeze()


class GaussianLayer(nn.Module):
    def __init__(self):
        super(GaussianLayer, self).__init__()

    def forward(self, x):
        if self.training:
            return x + torch.randn_like(x)
        return x


class Classifier(nn.Module):
    def __init__(self, i_dim, h_dim, nc, use_nfl = True, **kwargs):
        super(Classifier, self).__init__()

        print('use nfl?', use_nfl)

        self.use_nfl = use_nfl
        if use_nfl:

            cf = 1
            self.ad = AttentiveDensenet([h_dim//cf, h_dim//cf, h_dim//cf, h_dim//cf, h_dim//cf, h_dim//cf] + [h_dim, h_dim, h_dim, h_dim, h_dim],16,16,4, att_sparsity=None,attn_dropout=0.0).cuda()

            self.x1_b = nn.Sequential(nn.InstanceNorm2d(3),
                               nn.Conv2d(i_dim, h_dim//cf, 3, 1, 1),
                               nn.BatchNorm2d(h_dim//cf, momentum=0.99, eps=1e-3),
                               nn.LeakyReLU(0.1, inplace=True))

            self.x2_b = nn.Sequential(nn.Conv2d(h_dim//cf, h_dim//cf, 3, 1, 1),
                               nn.BatchNorm2d(h_dim//cf, momentum=0.99, eps=1e-3),
                               nn.LeakyReLU(0.1, inplace=True))

            self.x3_b = nn.Sequential(nn.Conv2d(h_dim//cf, h_dim//cf, 3, 1, 1),
                               nn.BatchNorm2d(h_dim//cf, momentum=0.99, eps=1e-3),
                               nn.LeakyReLU(0.1, inplace=True))

            self.x4_b = nn.Sequential(nn.MaxPool2d(2, 2),
                               nn.Dropout(0.5),
                               GaussianLayer(),
                               nn.Conv2d(h_dim//cf, h_dim//cf, 3, 1, 1),
                               nn.BatchNorm2d(h_dim//cf, momentum=0.99, eps=1e-3),
                               nn.LeakyReLU(0.1, inplace=True))

            self.x5_b = nn.Sequential(nn.Conv2d(h_dim//cf, h_dim//cf, 3, 1, 1),
                               nn.BatchNorm2d(h_dim//cf, momentum=0.99, eps=1e-3),
                               nn.LeakyReLU(0.1, inplace=True))

            self.x6_b = nn.Sequential(nn.Conv2d(h_dim//cf, h_dim//cf, 3, 1, 1),
                               nn.BatchNorm2d(h_dim//cf, momentum=0.99, eps=1e-3),
                               nn.LeakyReLU(0.1, inplace=True))

            self.x7_b = nn.Sequential(nn.MaxPool2d(2, 2),
                               nn.Dropout(0.5),
                               GaussianLayer())



        self.x1 = nn.Sequential(nn.InstanceNorm2d(3),
                               nn.Conv2d(i_dim, h_dim, 3, 1, 1),
                               nn.BatchNorm2d(h_dim, momentum=0.99, eps=1e-3),
                               nn.LeakyReLU(0.1, inplace=True))

        self.x2 = nn.Sequential(nn.Conv2d(h_dim, h_dim, 3, 1, 1),
                               nn.BatchNorm2d(h_dim, momentum=0.99, eps=1e-3),
                               nn.LeakyReLU(0.1, inplace=True))

        self.x3 = nn.Sequential(nn.Conv2d(h_dim, h_dim, 3, 1, 1),
                               nn.BatchNorm2d(h_dim, momentum=0.99, eps=1e-3),
                               nn.LeakyReLU(0.1, inplace=True))

        self.x4 = nn.Sequential(nn.MaxPool2d(2, 2),
                               nn.Dropout(0.5),
                               GaussianLayer(),
                               nn.Conv2d(h_dim, h_dim, 3, 1, 1),
                               nn.BatchNorm2d(h_dim, momentum=0.99, eps=1e-3),
                               nn.LeakyReLU(0.1, inplace=True))

        self.x5 = nn.Sequential(nn.Conv2d(h_dim, h_dim, 3, 1, 1),
                               nn.BatchNorm2d(h_dim, momentum=0.99, eps=1e-3),
                               nn.LeakyReLU(0.1, inplace=True))

        self.x6 = nn.Sequential(nn.Conv2d(h_dim, h_dim, 3, 1, 1),
                               nn.BatchNorm2d(h_dim, momentum=0.99, eps=1e-3),
                               nn.LeakyReLU(0.1, inplace=True))

        self.x7 = nn.Sequential(nn.MaxPool2d(2, 2),
                               nn.Dropout(0.5),
                               GaussianLayer())

        self.mlp = nn.Sequential(nn.Conv2d(h_dim, h_dim, 3, 1, 1),
                                 nn.BatchNorm2d(h_dim, momentum=0.99, eps=1e-3),
                                 nn.LeakyReLU(0.1, inplace=True),
                                 nn.Conv2d(h_dim, h_dim, 3, 1, 1),
                                 nn.BatchNorm2d(h_dim, momentum=0.99, eps=1e-3),
                                 nn.LeakyReLU(0.1, inplace=True),
                                 nn.Conv2d(h_dim, h_dim, 3, 1, 1),
                                 nn.BatchNorm2d(h_dim, momentum=0.99, eps=1e-3),
                                 nn.LeakyReLU(0.1, inplace=True),
                                 nn.AvgPool2d(8, 1),
                                 nn.Conv2d(h_dim, nc, 1, 1, 0),
                                 nn.BatchNorm2d(nc, momentum=0.99, eps=1e-3))
        
        self.init()

    def init(self):
        groups = [self.x1, self.x2, self.x3, self.x4, self.x5, self.x6, self.x7]
        if self.use_nfl:
            groups += [self.x1_b, self.x2_b, self.x3_b, self.x4_b, self.x5_b, self.x6_b, self.x7_b]
        
        for group in groups:
            for layer in group:
                if 'Conv2d' in layer.__class__.__name__:
                    nn.init.kaiming_normal_(layer.weight, mode='FAN_IN')
                    layer.bias.data.fill_(0)

    def forward(self, x, run_mlp=True):
        o = x*1.0

        if self.use_nfl:
            self.ad.reset()
            o = self.x1_b(o)
            o = self.x2_b(o)
            o = self.ad(o, write=True, read=False)
            o = self.x3_b(o)
            o = self.ad(o, write=True, read=False)
            o = self.x4_b(o)
            o = self.ad(o, write=True, read=False)
            o = self.x5_b(o)
            o = self.ad(o, write=True, read=False)
            o = self.x6_b(o)
            o = self.ad(o, write=True, read=False)
            o = self.x7_b(o)
            o = self.ad(o, write=True, read=False)
            o = x*1.0

        o = self.x1(o)
        if self.use_nfl:
            o = self.ad(o, write=True, read=True)
        o = self.x2(o)
        if self.use_nfl:
            o = self.ad(o, write=True, read=True)
        o = self.x3(o)
        if self.use_nfl:
            o = self.ad(o, write=True, read=True)
        o = o.contiguous()
        o = self.x4(o)
        if self.use_nfl:
            o = self.ad(o, write=True, read=True)
        o = self.x5(o)
        if self.use_nfl:
            o = self.ad(o, write=False, read=True)
        o = self.x6(o)
        o = o.contiguous()
        o = self.x7(o)
        o = o.contiguous()
        if run_mlp:
            o = self.mlp(o).squeeze()
        

        return o

