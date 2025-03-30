import torch.nn as nn

class SNGAN_Generator(nn.Module):
    def __init__(self, latent_dim=100, nc=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 384),
            nn.ReLU(),
            nn.Unflatten(1, (384, 1, 1)),
            nn.ConvTranspose2d(384, 192, 4, 1, 0),  # 1x1 → 4x4
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.ConvTranspose2d(192, 96, 4, 2, 1),   # 4x4 → 8x8
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.ConvTranspose2d(96, 48, 4, 2, 1),    # 8x8 → 16x16
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.ConvTranspose2d(48, nc, 4, 2, 1),    # 16x16 → 32x32
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

class SNGAN_Discriminator(nn.Module):
    def __init__(self, nc=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(nc, 16, 3, 2, 1),  # 32x32 → 16x16
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1),
        )

    def forward(self, x):
        return self.model(x)
