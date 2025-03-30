import torch
import torch.nn as nn

import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, img_channels=3):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(img_channels, 64, 4, stride=2, padding=1)),  # 28 → 14
            nn.LeakyReLU(0.2),

            nn.utils.spectral_norm(nn.Conv2d(64, 128, 4, stride=2, padding=1)),  # 14 → 7
            nn.LeakyReLU(0.2),

            nn.utils.spectral_norm(nn.Conv2d(128, 256, 3, stride=2, padding=1)),  # 7 → 4
            nn.LeakyReLU(0.2),

            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(256 * 4 * 4, 1))
            # No Sigmoid!
        )

    def forward(self, x):
        return self.net(x)



    
class Generator(nn.Module):
    def __init__(self, z_dim=100, img_channels=3):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256 * 7 * 7),
            nn.BatchNorm1d(256 * 7 * 7),
            nn.ReLU(True),

            nn.Unflatten(1, (256, 7, 7)),

            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 7x7 → 14x14
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 14x14 → 28x28
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, img_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # Output in [-1, 1]
        )

    def forward(self, z):
        return self.net(z)
