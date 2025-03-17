import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, img_channels=3):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(img_channels, 64, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.1),

            nn.utils.spectral_norm(nn.Conv2d(64, 128, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.1),
            nn.Flatten(),

            nn.utils.spectral_norm(nn.Linear(128*7*7, 1)),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.net(img)
    

class Generator(nn.Module):
    def __init__(self, z_dim, img_channels=3):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 128*7*7),
            nn.ReLU(),
            nn.Unflatten(1, (128, 7, 7)),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, img_channels, 4, stride=2, padding=1),
            nn.Tanh()     
        )

    def forward(self, z):
        return self.net(z)

        