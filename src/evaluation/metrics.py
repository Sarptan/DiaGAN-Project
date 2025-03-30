import torch
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance



def preprocess_images(images):
    """
    Ensure images are 3-channel and in [0, 1]
    """
    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)
    images = (images + 1) / 2  # [-1, 1] -> [0, 1]
    return images.clamp(0, 1)


def compute_inception_score(images, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    images = preprocess_images(images).to(device).to(torch.uint8) * 255
    metric = InceptionScore().to(device)
    return metric(images)[0].item(), metric(images)[1].item()


def compute_fid(real_images, fake_images, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    real_images = preprocess_images(real_images).to(device)
    fake_images = preprocess_images(fake_images).to(device)
    metric = FrechetInceptionDistance(normalize=True).to(device)
    metric.update(real_images, real=True)
    metric.update(fake_images, real=False)
    return metric.compute().item()


def compute_precision_recall(real_images, generated_images, k=3, eps=0.1):
    """
    Compare precision and recall between real and generated images using k-NN.

    Args:
        real_images: numpy array or torch tensor, shape (N, C, H, W) or (N, H, W) or (N, D)
        generated_images: same format
        k: number of neighbors in k-NN
        eps: distance threshold for membership (default 0.1)

    Returns:
        precision, recall
    """
    import numpy as np
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import normalize

    def preprocess(images):
        # Convert torch tensors to numpy
        if 'torch' in str(type(images)):
            images = images.detach().cpu().numpy()

        # Ensure float32 dtype
        images = images.astype(np.float32)

        # Flatten to (N, D)
        if images.ndim == 4:  # (N, C, H, W)
            images = images.reshape(images.shape[0], -1)
        elif images.ndim == 3:  # (N, H, W)
            images = images.reshape(images.shape[0], -1)
        elif images.ndim == 2:
            pass  # already (N, D)
        else:
            raise ValueError(f"Unsupported image shape: {images.shape}")

        return normalize(images)

    real_vecs = preprocess(real_images)
    gen_vecs = preprocess(generated_images)

    # k-NN on real
    nn_real = NearestNeighbors(n_neighbors=k).fit(real_vecs)
    dists_real_to_gen, _ = nn_real.kneighbors(gen_vecs)
    precision = np.mean(np.min(dists_real_to_gen, axis=1) < eps)

    # k-NN on generated
    nn_gen = NearestNeighbors(n_neighbors=k).fit(gen_vecs)
    dists_gen_to_real, _ = nn_gen.kneighbors(real_vecs)
    recall = np.mean(np.min(dists_gen_to_real, axis=1) < eps)

    return precision, recall


from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.metrics import pairwise_distances


def compute_precision_recall_2D(real_samples, fake_samples, k=5, eps=1e-8):
    """
    Compute precision and recall in 2D space using kNN radii.
    - Precision: % of fake samples close to some real samples
    - Recall:    % of real samples close to some fake samples
    """
    # kNN radius in real space (for recall)
    nn_real = NearestNeighbors(n_neighbors=k).fit(real_samples)
    dists_real, _ = nn_real.kneighbors(real_samples)
    real_radii = dists_real[:, -1] + eps

    # kNN radius in fake space (for precision)
    nn_fake = NearestNeighbors(n_neighbors=k).fit(fake_samples)
    dists_fake, _ = nn_fake.kneighbors(fake_samples)
    fake_radii = dists_fake[:, -1] + eps

    # Recall: how many real samples fall within fake neighborhoods
    dists_rf = pairwise_distances(real_samples, fake_samples)
    recall = np.mean(np.min(dists_rf, axis=1) <= fake_radii[np.argmin(dists_rf, axis=1)])

    # Precision: how many fake samples fall within real neighborhoods
    dists_fr = pairwise_distances(fake_samples, real_samples)
    precision = np.mean(np.min(dists_fr, axis=1) <= real_radii[np.argmin(dists_fr, axis=1)])

    return precision, recall

def evaluate_mode_coverage(fake_samples, real_centers, threshold=0.1, min_samples_per_mode=10):
    """
    Counts how many real modes are covered by fake samples.
    A mode is 'covered' if at least `min_samples_per_mode` fake samples are within `threshold` distance.
    """
    covered = 0
    for center in real_centers:
        distances = np.linalg.norm(fake_samples - center, axis=1)
        n_close = np.sum(distances < threshold)
        if n_close >= min_samples_per_mode:
            covered += 1
    return covered