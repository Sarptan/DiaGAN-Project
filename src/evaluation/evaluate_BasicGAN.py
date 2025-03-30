import torch
from utils.plot import plot_batch_images
import numpy as np
import torch
from scipy.linalg import sqrtm
from evaluation.metrics import compute_precision_recall


def generate_BasicGAN(generator, latent_dim=100, num_samples=16, plotit=False):
    """
    Generates images using a trained generator and optionally plots them.
    
    Args:
        generator (nn.Module): Trained generator model.
        latent_dim (int): Dimension of the latent space.
        num_samples (int): Number of images to generate.
        plotit (bool): If True, plots the generated images.
    
    Returns:
        torch.Tensor: Generated images of shape (num_samples, C, H, W).
    """
    generator.eval()  # Set generator to evaluation mode
    
    with torch.no_grad():  # No gradients needed for evaluation
        z = torch.randn(num_samples, latent_dim)  # Random latent vectors
        generated_images = generator(z)  # Generate images
    
    if plotit:
        plot_batch_images(generated_images, None)
    
    return generated_images


def get_classifier_features(classifier, data, batch_size=128):
    classifier.eval()
    device = next(classifier.parameters()).device
    features = []
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = torch.tensor(data[i:i+batch_size], dtype=torch.float32, device=device)
            if batch.ndim == 3:
                batch = batch.unsqueeze(1)
            _, feats = classifier(batch)
            features.append(feats.cpu())
    return torch.cat(features, dim=0).numpy()


def compute_fid(real_feats, fake_feats):
    mu1, mu2 = real_feats.mean(0), fake_feats.mean(0)
    sigma1 = np.cov(real_feats, rowvar=False)
    sigma2 = np.cov(fake_feats, rowvar=False)
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = np.sum((mu1 - mu2) ** 2) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

def evaluate_gan(gen_images, classifier, real_data, device="cpu"):
    classifier = classifier.to(device)

    fake = gen_images
    
    # Ensure fake has shape (N, 1, 28, 28)
    if fake.ndim == 3:
        fake = fake[:, None, :, :]
    elif fake.shape[1] != 1:
        fake = fake[:, :1, :, :]

    real_feats = get_classifier_features(classifier, real_data)
    fake_feats = get_classifier_features(classifier, fake)

    prec, rec = compute_precision_recall(real_data, fake,eps=0.4)
    fid = compute_fid(real_feats, fake_feats)

    return {"Precision": prec, "Recall": rec, "FID": fid}