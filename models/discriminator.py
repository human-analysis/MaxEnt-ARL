# discriminator.py

import numpy as np
import torch.nn as nn


def create_discriminator(embed_length=512, num_classes=10, type=1):

    if type == 1:
        return Discriminator1(embed_length=embed_length, num_classes=num_classes)
    elif type == 2:
        return Discriminator2(embed_length=embed_length, num_classes=num_classes)
    elif type == 3:
        return Discriminator3(embed_length=embed_length, num_classes=num_classes)
    elif type == 4:
        return Discriminator4(embed_length=embed_length, num_classes=num_classes)
    elif type == 5:
        return Discriminator5(embed_length=embed_length, num_classes=num_classes)
    elif type == 6:
        return Discriminator6(embed_length=embed_length, num_classes=num_classes)
    elif type == 7:
        return Discriminator7(embed_length=embed_length, num_classes=num_classes)
    elif type == 8:
        return Discriminator8(embed_length=embed_length, num_classes=num_classes)
    elif type == 9:
        return Discriminator9(embed_length=embed_length, num_classes=num_classes)
    elif type == 10:
        return Discriminator10(embed_length=embed_length, num_classes=num_classes)
    elif type == 11:
        return Discriminator11(embed_length=embed_length, num_classes=num_classes)
    else:
        raise Exception("Discriminator not supported")


class DiscriminatorMNIST(nn.Module):
    def __init__(self, opt):
        self.opt = opt
        super(DiscriminatorMNIST, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.opt.img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity


class Discriminator1(nn.Module):

    def __init__(self, embed_length, num_classes=10):
        super().__init__()
        self.classlayer = nn.Linear(embed_length, num_classes)

    def forward(self, x):
        z = x
        out = self.classlayer(x)

        return out, z


class Discriminator2(nn.Module):

    def __init__(self, embed_length, num_classes=10):
        super().__init__()

        self.model1 = nn.Sequential(
            nn.Linear(embed_length, 256),
            nn.BatchNorm1d(256),
            nn.PReLU(),
            nn.Dropout()
        )
        self.classlayer = nn.Linear(256, num_classes)
        self.softmaxlayer = nn.Softmax(dim=1)

    def forward(self, x):

        z = self.model1(x)
        out = self.classlayer(z)
        prob = self.softmaxlayer(out)

        return out, z, prob


class Discriminator3(nn.Module):
    def __init__(self, embed_length, num_classes=100):
        super().__init__()

        self.model1 = nn.Sequential(
            nn.Linear(embed_length, 256),
            nn.BatchNorm1d(256),
            nn.PReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.PReLU(),
        )
        self.classlayer = nn.Linear(128, num_classes)
        self.softmaxlayer = nn.Softmax(dim=1)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        z = self.model1(x)
        out = self.classlayer(z)
        prob = self.softmaxlayer(out) + 1e-16
        out = self.logsoftmax(out)
        return out, z, prob


class Discriminator4(nn.Module):
    def __init__(self, embed_length=128, num_classes=1000):
        super().__init__()

        self.model1 = nn.Sequential(
            nn.Linear(embed_length, 128),
            nn.BatchNorm1d(128),
            nn.PReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.PReLU(),
        )
        self.classlayer = nn.Linear(64, num_classes)
        self.softmaxlayer = nn.Softmax(dim=1)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        z = self.model1(x)
        out = self.classlayer(z)
        prob = self.softmaxlayer(out) + 1e-16
        out = self.logsoftmax(out)

        return out, z, prob


class Discriminator5(nn.Module):
    def __init__(self, embed_length, num_classes=10):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(embed_length, 20),
            nn.BatchNorm1d(20),
            nn.PReLU(),
            nn.Linear(20, 20),
            nn.BatchNorm1d(20),
            nn.PReLU(),
            nn.Dropout()
        )
        self.classlayer = nn.Linear(20, num_classes)

    def forward(self, x):
        z = self.model(x)
        out = self.classlayer(z)

        return out, z


class Discriminator6(nn.Module):
    def __init__(self, embed_length, num_classes=10):
        super().__init__()

        self.model1 = nn.Sequential(
            nn.Linear(embed_length, 32),
            nn.BatchNorm1d(32),
            nn.PReLU(),
        )
        self.classlayer = nn.Linear(32, num_classes)

    def forward(self, x):
        z = self.model1(x)

        out = self.classlayer(z)

        return out, z


class Discriminator7(nn.Module):
    def __init__(self, embed_length, num_classes=10):
        super().__init__()

        self.model1 = nn.Sequential(
            nn.Linear(embed_length, 256),
            nn.BatchNorm1d(256),
            nn.PReLU(),
            nn.Dropout(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.PReLU(),
            nn.Dropout(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.PReLU(),
            nn.Dropout()
        )
        self.classlayer = nn.Linear(64, num_classes)

    def forward(self, x):
        z = self.model1(x)

        out = self.classlayer(z)

        return out, z


class DiscriminatorN(nn.Module):
    def __init__(self, embed_length, num_classes=10):
        super().__init__()

        self.model1 = nn.Sequential(
            nn.Linear(embed_length, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )
        self.model2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.ReLU(),
        )
        self.shortcut = nn.Sequential(nn.Linear(512, 512))
        self.classlayer = nn.Linear(2, num_classes)

    def forward(self, x):
        z = self.model1(x)
        z += self.shortcut(x)
        z = self.model2(z)
        out = self.classlayer(z)

        return out, z


class Discriminator8(nn.Module):
    def __init__(self, embed_length, num_classes=10):
        super().__init__()

        self.model1 = nn.Sequential(
            nn.Linear(embed_length, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )
        self.model2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.ReLU(),
        )
        self.shortcut = nn.Sequential(nn.Linear(512, 512))
        self.classlayer = nn.Linear(2, num_classes)

    def forward(self, x):
        z = self.model1(x)
        z += self.shortcut(x)
        z = self.model2(z)
        out = self.classlayer(z)

        return out, z


class Discriminator9(nn.Module):

    def __init__(self, embed_length=64, num_classes=10):
        super().__init__()

        self.model1 = nn.Sequential(
            nn.Linear(embed_length, 32),
            nn.BatchNorm1d(32),
            nn.PReLU(),
            nn.Dropout()
        )
        self.classlayer = nn.Linear(32, num_classes)

    def forward(self, x):

        z = self.model1(x)
        out = self.classlayer(z)

        return out, z


class Discriminator10(nn.Module):
    def __init__(self, embed_length=512, num_classes=10):
        super().__init__()

        self.model1 = nn.Sequential(
            nn.Linear(embed_length, 256),
            nn.BatchNorm1d(256),
            nn.PReLU(),
            nn.Dropout(),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.PReLU(),
            nn.Dropout()
        )
        self.classlayer = nn.Linear(64, num_classes)

    def forward(self, x):
        z = self.model1(x)

        out = self.classlayer(z)

        return out, z


class Discriminator11(nn.Module):
    def __init__(self, embed_length=2, num_classes=10):
        super().__init__()

        self.model1 = nn.Sequential(
            nn.Linear(embed_length, 2),
            nn.ReLU(),
            nn.Linear(2, 2),
            nn.ReLU(),
        )
        self.classlayer = nn.Linear(2, num_classes)

    def forward(self, x):
        z = self.model1(x)
        out = self.classlayer(z)

        return out, z
