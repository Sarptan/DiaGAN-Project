import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import sqrtm



def plot_generated_colored_images(generator, latent_dim=100, num_images=16, device="cpu", title="Generated Samples", plot_it=False):
    """
  
    """
    generator.eval()
    z = torch.randn(num_images, latent_dim).to(device)
    with torch.no_grad():
        fake_imgs = generator(z).cpu()

    # Ensure range [0, 1] for visualization
    fake_imgs = (fake_imgs + 1) / 2  # Assuming Tanh output → scale to [0, 1]

    if plot_it:
        # Plot in grid
        nrow = int(np.sqrt(num_images))
        grid_img = torchvision.utils.make_grid(fake_imgs, nrow=nrow, normalize=False)
        plt.figure(figsize=(8, 8))
        plt.imshow(grid_img.permute(1, 2, 0))  # (C, H, W) → (H, W, C)
        plt.axis("off")
        plt.title(title)
        plt.show()

    return fake_imgs



def get_classifier_features(classifier, data, batch_size=128):
    classifier.eval()
    device = next(classifier.parameters()).device
    features = []
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = torch.tensor(data[i:i+batch_size], dtype=torch.float32, device=device)

            if batch.ndim == 3:
                batch = batch.unsqueeze(1)  # (N, H, W) → (N, 1, H, W)

            if batch.shape[1] == 1 and next(classifier.parameters()).shape[1] == 3:
                batch = batch.repeat(1, 3, 1, 1)  # grayscale → RGB for classifier

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

def compute_precision_recall(real_images, generated_images, k=3, eps=0.1):
    import numpy as np
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import normalize

    def preprocess(images):
        if 'torch' in str(type(images)):
            images = images.detach().cpu().numpy()

        images = images.astype(np.float32)

        if images.ndim == 4:  # (N, C, H, W)
            images = images.reshape(images.shape[0], -1)
        elif images.ndim == 3:  # (N, H, W)
            images = images.reshape(images.shape[0], -1)
        elif images.ndim == 2:
            pass
        else:
            raise ValueError(f"Unsupported image shape: {images.shape}")

        return normalize(images)

    real_vecs = preprocess(real_images)
    gen_vecs = preprocess(generated_images)

    nn_real = NearestNeighbors(n_neighbors=k).fit(real_vecs)
    dists_real_to_gen, _ = nn_real.kneighbors(gen_vecs)
    precision = np.mean(np.min(dists_real_to_gen, axis=1) < eps)

    nn_gen = NearestNeighbors(n_neighbors=k).fit(gen_vecs)
    dists_gen_to_real, _ = nn_gen.kneighbors(real_vecs)
    recall = np.mean(np.min(dists_gen_to_real, axis=1) < eps)

    return precision, recall



def evaluate_gan(gen_images, classifier, real_data, device="cpu", eps=0.4):
    classifier = classifier.to(device)
    
    # Ensure correct shape and channel match
    def to_tensor(imgs):
        imgs = torch.tensor(imgs, dtype=torch.float32)
        if imgs.ndim == 3:
            imgs = imgs.unsqueeze(1)  # (N, H, W) → (N, 1, H, W)
        return imgs

    real_tensor = to_tensor(real_data)
    fake_tensor = to_tensor(gen_images)

    # Match channels for pixel-based metrics
    if real_tensor.shape[1] != fake_tensor.shape[1]:
        if real_tensor.shape[1] == 1:
            real_tensor = real_tensor.repeat(1, 3, 1, 1)
        if fake_tensor.shape[1] == 1:
            fake_tensor = fake_tensor.repeat(1, 3, 1, 1)

    # Feature extraction for FID
    real_feats = get_classifier_features(classifier, real_tensor)
    fake_feats = get_classifier_features(classifier, fake_tensor)

    # Pixel-space precision/recall (optional)
    prec, rec = compute_precision_recall(real_feats, fake_feats, eps=eps)
    fid = compute_fid(real_feats, fake_feats)

    return {"Precision": prec, "Recall": rec, "FID": fid}
