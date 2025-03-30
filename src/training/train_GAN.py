import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, WeightedRandomSampler, TensorDataset
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def train_Basic2DGAN(generator, discriminator, dataloader, lr=0.0003, latent_dim=2, n_epochs=200, plotit=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    criterion = nn.BCELoss()
    g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    g_losses, d_losses = [], []

    for epoch in range(n_epochs):
        for real_data, _ in dataloader:
            real_data = real_data.to(device)
            batch_size = real_data.size(0)

            # --- Real and fake labels ---
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # --- Train Discriminator ---
            z = torch.randn(batch_size, latent_dim, device=device)
            fake_data = generator(z)

            d_real = discriminator(real_data)
            d_fake = discriminator(fake_data.detach())

            d_loss_real = criterion(d_real, real_labels)
            d_loss_fake = criterion(d_fake, fake_labels)
            d_loss = d_loss_real + d_loss_fake

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # --- Train Generator ---
            z = torch.randn(batch_size, latent_dim, device=device)
            fake_data = generator(z)
            d_pred = discriminator(fake_data)
            g_loss = criterion(d_pred, real_labels)

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

        g_losses.append(g_loss.item())
        d_losses.append(d_loss.item())



def compute_discrepancy_score(ldrs, window_size=50, k=1.0):
    ldrs = np.array(ldrs[-window_size:])  # Use last `window_size` values
    if len(ldrs) == 0:
        return 1.0  # fallback
    ldrm = np.mean(ldrs)
    ldrv = np.var(ldrs)
    return ldrm + k * np.sqrt(ldrv)




def train_BasicGAN(generator, discriminator, dataloader, lr=0.0001, criterion=None,
                   latent_dim: int = 100, n_epochs: int = 20, plotit=False):
    

    if criterion is None:
        criterion = nn.BCEWithLogitsLoss() 

    d_losses, g_losses = [], []
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(n_epochs):
        for i, (imgs, _) in enumerate(dataloader):
            batch_size = imgs.size(0)

            real_labels = torch.ones(batch_size, 1)
            fake_labels = torch.zeros(batch_size, 1)

            z = torch.randn(batch_size, latent_dim)
            fake_imgs = generator(z)

            # Train Discriminator
            d_optimizer.zero_grad()
            real_loss = criterion(discriminator(imgs), real_labels)
            fake_loss = criterion(discriminator(fake_imgs.detach()), fake_labels)
            d_loss = real_loss + fake_loss
            d_loss.backward()
            d_optimizer.step()

            # Train Generator
            g_optimizer.zero_grad()
            g_loss = criterion(discriminator(fake_imgs), real_labels)
            g_loss.backward()
            g_optimizer.step()

        d_losses.append(d_loss.item())
        g_losses.append(g_loss.item())
        #print(f"Epoch [{epoch+1}/{n_epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

    if plotit:
        plt.figure(figsize=(10, 5))
        plt.plot(d_losses, label="Discriminator Loss")
        plt.plot(g_losses, label="Generator Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()




import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, TensorDataset
import numpy as np
from collections import defaultdict

def train_DiaGAN(generator, discriminator, data, lr=0.0005, latent_dim=100, n_epochs=200,
                 phase1_ratio=0.9, batch_size=128, window_size=50, k=1.0,
                 min_clip=0.01, max_ratio=50):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    discriminator.to(device)

    criterion = nn.BCEWithLogitsLoss()  # use logits (SNGAN-friendly)

    # Detect if image data (e.g., MNIST shape: [B, 1, 28, 28])
    is_image = (data.ndim > 2)
    tensor_data = torch.tensor(data, dtype=torch.float32)

    # Detect if discriminator expects 4D conv inputs
    conv_mode = any(isinstance(m, nn.Conv2d) for m in discriminator.modules())

    dataset_size = len(tensor_data)
    ldr_dict = defaultdict(list)

    for epoch in range(n_epochs):
        in_phase1 = epoch < int(phase1_ratio * n_epochs)

        #Phase 1: uniform sampling
        if in_phase1:
            sampler = torch.utils.data.RandomSampler(tensor_data)

        #Phase 2: weighted sampling by LDRM + k * sqrt(LDRV)
        else:
            scores = []
            for i in range(dataset_size):
                ldrs = np.array(ldr_dict[i][-window_size:])
                if len(ldrs) == 0:
                    scores.append(1.0)
                else:
                    ldrm = np.mean(ldrs)
                    ldrv = np.var(ldrs)
                    score = ldrm + k * np.sqrt(ldrv)
                    scores.append(score)

            scores = np.clip(scores, a_min=min_clip, a_max=min_clip * max_ratio)
            probs = scores / scores.sum()
            sampler = WeightedRandomSampler(probs, num_samples=dataset_size, replacement=True)

        dataloader = DataLoader(TensorDataset(tensor_data), batch_size=batch_size, sampler=sampler)

        for real_batch, in dataloader:
            real_imgs = real_batch.to(device)
            bs = real_imgs.size(0)

            real_labels = torch.ones(bs, 1, device=device)
            fake_labels = torch.zeros(bs, 1, device=device)

            z = torch.randn(bs, latent_dim, device=device)
            fake_imgs = generator(z)

            # Choose input shape depending on model type
            if conv_mode:
                real_input = real_imgs
                fake_input = fake_imgs
            else:
                real_input = real_imgs.view(bs, -1)
                fake_input = fake_imgs.view(bs, -1)

            # Discriminator step
            d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
            d_optimizer.zero_grad()
            real_out = discriminator(real_input)
            fake_out = discriminator(fake_input.detach())
            d_loss = criterion(real_out, real_labels) + criterion(fake_out, fake_labels)
            d_loss.backward()
            d_optimizer.step()

            # Generator step
            g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
            g_optimizer.zero_grad()
            fake_out = discriminator(fake_input)
            g_loss = criterion(fake_out, real_labels)
            g_loss.backward()
            g_optimizer.step()

        # --- Track LDR values ---
        with torch.no_grad():
            D_x = discriminator(tensor_data.to(device) if not conv_mode else tensor_data.to(device)).squeeze()
            eps = 1e-6
            D_x_sigmoid = torch.sigmoid(D_x)  # for LDR calc even tho BCEWithLogits used
            LDR_x = torch.log(D_x_sigmoid / (1 - D_x_sigmoid + eps)).cpu().numpy()
            for i in range(dataset_size):
                ldr_dict[i].append(LDR_x[i])

        if epoch % 10 == 0:
            print(f"[Epoch {epoch}] D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

    return generator


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, TensorDataset
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

def train_Dia2DGAN(generator, discriminator, data, lr=0.0003, latent_dim=2, n_epochs=200,
                   phase1_ratio=0.9, batch_size=128, window_size=50, k=0.3,
                   min_clip=0.01, max_ratio=50, plotit=False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    discriminator.to(device)

    criterion = nn.BCELoss()  # Expect output in [0, 1] from discriminator
    g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    tensor_data = torch.tensor(data, dtype=torch.float32)
    dataset_size = len(tensor_data)
    ldr_dict = defaultdict(list)

    g_losses, d_losses = [], []

    for epoch in range(n_epochs):
        in_phase1 = epoch < int(phase1_ratio * n_epochs)

        if in_phase1:
            sampler = torch.utils.data.RandomSampler(tensor_data)
        else:
            scores = []
            for i in range(dataset_size):
                ldrs = np.array(ldr_dict[i][-window_size:])
                if len(ldrs) == 0:
                    scores.append(1.0)
                else:
                    ldrm = np.mean(ldrs)
                    ldrv = np.var(ldrs)
                    score = ldrm + k * np.sqrt(ldrv)
                    scores.append(score)

            scores = np.clip(scores, a_min=min_clip, a_max=min_clip * max_ratio)
            probs = scores / scores.sum()
            sampler = WeightedRandomSampler(probs, num_samples=dataset_size, replacement=True)

        dataloader = DataLoader(TensorDataset(tensor_data), batch_size=batch_size, sampler=sampler)

        for real_batch, in dataloader:
            real_imgs = real_batch.to(device)
            bs = real_imgs.size(0)

            real_labels = torch.ones(bs, 1, device=device)
            fake_labels = torch.zeros(bs, 1, device=device)

            z = torch.randn(bs, latent_dim, device=device)
            fake_imgs = generator(z)

            # Train Discriminator
            d_optimizer.zero_grad()
            d_real = discriminator(real_imgs)
            d_fake = discriminator(fake_imgs.detach())
            d_loss = criterion(d_real, real_labels) + criterion(d_fake, fake_labels)
            d_loss.backward()
            d_optimizer.step()

            # Train Generator
            g_optimizer.zero_grad()
            g_pred = discriminator(fake_imgs)
            g_loss = criterion(g_pred, real_labels)
            g_loss.backward()
            g_optimizer.step()

        # Track LDR(x) = log(D / (1 - D))
        with torch.no_grad():
            d_out = discriminator(tensor_data.to(device)).squeeze()
            ldr = torch.log(d_out / (1 - d_out + 1e-6)).cpu().numpy()
            for i in range(dataset_size):
                ldr_dict[i].append(ldr[i])

        g_losses.append(g_loss.item())
        d_losses.append(d_loss.item())

        if epoch % 50 == 0:
            print(f"[Epoch {epoch}] D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")
            if plotit:
                with torch.no_grad():
                    z = torch.randn(1000, latent_dim, device=device)
                    gen_samples = generator(z).cpu().numpy()
                    plt.figure(figsize=(4, 4))
                    plt.scatter(gen_samples[:, 0], gen_samples[:, 1], s=2, alpha=0.5)
                    plt.title(f"Generated Samples @ Epoch {epoch}")
                    plt.axis('equal')
                    plt.grid(True)
                    plt.show()

    return generator

from models import SNGAN
def apply_DRS(generator, real_data, z_dim=100, num_gen=10000, batch_size=128, n_epochs=5):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare training data
    real_tensor = torch.tensor(real_data, dtype=torch.float32)
    real_labels = torch.ones(len(real_tensor), 1)
    real_ds = TensorDataset(real_tensor, real_labels)

    # Generate fake data
    generator.eval()
    with torch.no_grad():
        z = torch.randn(num_gen, z_dim, device=device)
        fake_tensor = generator(z).detach().cpu()
    fake_labels = torch.zeros(len(fake_tensor), 1)
    fake_ds = TensorDataset(fake_tensor, fake_labels)

    # Combine real and fake
    combined_data = torch.cat([real_tensor, fake_tensor], dim=0)
    combined_labels = torch.cat([real_labels, fake_labels], dim=0)
    loader = DataLoader(TensorDataset(combined_data, combined_labels), batch_size=batch_size, shuffle=True)

    # Train auxiliary discriminator
    aux_disc = SNGAN.Discriminator(img_channels=1).to(device) 
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(aux_disc.parameters(), lr=0.0002, betas=(0.5, 0.999))

    print("Training auxiliary discriminator for DRS...")
    aux_disc.train()
    for epoch in range(n_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = aux_disc(x)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"[Epoch {epoch+1}] Aux D Loss: {loss.item():.4f}")
    aux_disc.eval()

    with torch.no_grad():
        logits = aux_disc(fake_tensor.to(device)).squeeze()
        probs = torch.sigmoid(logits)
        ldr = probs / (1 - probs + 1e-6)

    # Normalize and resample with probability LDR
    ldr_np = ldr.cpu().numpy()
    acceptance_probs = ldr_np / ldr_np.max()  # scale to [0, 1]
    accept_flags = np.random.rand(len(ldr_np)) < acceptance_probs

    filtered_samples = fake_tensor[accept_flags]
    print(f"DRS Accepted {len(filtered_samples)}/{num_gen} samples ({accept_flags.mean()*100:.2f}%)")
    return filtered_samples.numpy()