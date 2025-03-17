import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


def train_BasicGAN(generator, discriminator, dataloader, lr=0.0001, criterion=nn.BCELoss(), 
          latent_dim:int=100, n_epochs:int=20, plotit=False):
    
    d_losses, g_losses = [], []
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(n_epochs):
        for i, (imgs,_) in enumerate(dataloader):
            batch_size = imgs.size(0)

            real_labels = torch.ones(batch_size, 1)
            fake_labels = torch.zeros(batch_size, 1)
            z = torch.randn(batch_size, latent_dim)
            fake_imgs  = generator(z)

            #train Discriminator
            d_optimizer.zero_grad()
            real_loss = criterion(discriminator(imgs), real_labels)
            fake_loss = criterion(discriminator(fake_imgs.detach()), fake_labels)
            d_loss = real_loss + fake_loss
            d_loss.backward()
            d_optimizer.step()

            #train Gemerator
            g_optimizer.zero_grad()
            g_loss = criterion(discriminator(fake_imgs), real_labels)
            g_loss.backward()
            g_optimizer.step()
        
        d_losses.append(d_loss.item())
        g_losses.append(g_loss.item())
        print(f"Epoch [{epoch+1}/{n_epochs}] | D Loss Basic: {d_loss.item():.4f} | G Loss Basic: {g_loss.item():.4f}")

    
    if plotit:
        plt.figure(figsize=(10,5))
        plt.plot(d_losses, label="Discriminator")
        plt.plot(g_losses, label="Gnerator")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()





