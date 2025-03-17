import torch
from utils.plot import plot_batch_images

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