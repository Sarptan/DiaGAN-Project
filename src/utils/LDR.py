import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import DataLoader

from utils.data_loader import get_colored_mnist_dataloader

def ldrm_ldrv_inspection(generator, discriminator, mnist_dataset, 
                         minority_ratios=[0.01, 0.05, 0.1, 0.2], 
                         latent_dim=100, n_epochs=5, batch_size=128):
    """
    Trains the GAN for different minority ratios and computes LDRM & LDRV for major and minor groups.
    
    Args:
        generator (nn.Module): The GAN generator.
        discriminator (nn.Module): The GAN discriminator.
        mnist_dataset (torchvision.datasets.MNIST): Preloaded MNIST dataset.
        minority_ratios (list): List of minority ratios to evaluate.
        latent_dim (int): Size of the latent noise vector.
        n_epochs (int): Number of training epochs for each ratio.
        batch_size (int): Batch size for training.
    
    Returns:
        Plots LDRM & LDRV vs. Minority Ratio.
    """
    
    ldrm_major_values, ldrm_minor_values = [], []
    ldrv_major_values, ldrv_minor_values = [], []

    criterion = torch.nn.BCELoss()
    
    for ratio in minority_ratios:
        print(f"\n Training GAN with Minority Ratio: {ratio:.2%}")

        # Create DataLoader with given minority ratio
        dataloader = get_colored_mnist_dataloader(mnist_dataset, batch_size=batch_size, minority_ratio=ratio)
        
        # Optimizers
        d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
        g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
        
        # Store LDR values per sample
        ldr_values = {}

        # Training loop
        for epoch in range(n_epochs):
            for imgs, _ in dataloader:
                batch_size = imgs.size(0)

                real_labels = torch.ones(batch_size, 1).to(imgs.device)
                fake_labels = torch.zeros(batch_size, 1).to(imgs.device)

                # Generate fake images
                z = torch.randn(batch_size, latent_dim).to(imgs.device)
                fake_imgs = generator(z)

                # Train Discriminator
                d_optimizer.zero_grad()
                real_preds = discriminator(imgs)
                fake_preds = discriminator(fake_imgs.detach())

                real_loss = criterion(real_preds, real_labels)
                fake_loss = criterion(fake_preds, fake_labels)
                d_loss = real_loss + fake_loss
                d_loss.backward()
                d_optimizer.step()

                # Train Generator
                g_optimizer.zero_grad()
                g_loss = criterion(discriminator(fake_imgs), real_labels)
                g_loss.backward()
                g_optimizer.step()

                # Store LDR values
                with torch.no_grad():
                    D_x = real_preds.sigmoid()
                    LDR_x = torch.log(D_x / (1 - D_x))  # Compute LDR

                    for j, sample in enumerate(imgs):
                        sample_id_hash = hash(sample.cpu().numpy().tobytes())  # Unique ID for each sample
                        if sample_id_hash not in ldr_values:
                            ldr_values[sample_id_hash] = {"LDR": [], "color": "major" if torch.any(sample[1] > 0) else "minor"}
                        ldr_values[sample_id_hash]["LDR"].append(LDR_x[j].item())

        # Compute LDRM and LDRV per group
        ldrm_major = np.mean([np.mean(v["LDR"]) for v in ldr_values.values() if v["color"] == "major"])
        ldrm_minor = np.mean([np.mean(v["LDR"]) for v in ldr_values.values() if v["color"] == "minor"])
        ldrv_major = np.mean([np.var(v["LDR"]) for v in ldr_values.values() if v["color"] == "major"])
        ldrv_minor = np.mean([np.var(v["LDR"]) for v in ldr_values.values() if v["color"] == "minor"])

        ldrm_major_values.append(ldrm_major)
        ldrm_minor_values.append(ldrm_minor)
        ldrv_major_values.append(ldrv_major)
        ldrv_minor_values.append(ldrv_minor)

        print(f" LDRM (Major): {ldrm_major:.4f} | LDRM (Minor): {ldrm_minor:.4f}")
        print(f" LDRV (Major): {ldrv_major:.4f} | LDRV (Minor): {ldrv_minor:.4f}")

    # Plot LDRM vs. Minority Ratio
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(minority_ratios, ldrm_major_values, marker="o", label="LDRM - Major (Red)")
    plt.plot(minority_ratios, ldrm_minor_values, marker="s", label="LDRM - Minor (Green)", linestyle="dashed")
    plt.xlabel("Minority Ratio")
    plt.ylabel("LDRM (Mean Log-Density Ratio)")
    plt.title("LDRM vs. Minority Ratio")
    plt.legend()
    
    # Plot LDRV vs. Minority Ratio
    plt.subplot(1, 2, 2)
    plt.plot(minority_ratios, ldrv_major_values, marker="o", label="LDRV - Major (Red)")
    plt.plot(minority_ratios, ldrv_minor_values, marker="s", label="LDRV - Minor (Green)", linestyle="dashed")
    plt.xlabel("Minority Ratio")
    plt.ylabel("LDRV (Variance of Log-Density Ratio)")
    plt.title("LDRV vs. Minority Ratio")
    plt.legend()

    plt.show()