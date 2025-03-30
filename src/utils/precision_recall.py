# precision_recall.py

import torch
import torch.nn as nn
from torchvision.models import inception_v3
import torch.nn.functional as F
from torchvision.transforms import Resize

class InceptionFeatureExtractor(nn.Module):
    def __init__(self, resize=True, device=None):
        super().__init__()
        self.resize = resize
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = inception_v3(pretrained=True, transform_input=False).to(self.device)
        self.model.eval()
        self.model.fc = nn.Identity()  # remove final classifier
        self.resize_op = Resize((299, 299)) if resize else nn.Identity()

    @torch.no_grad()
    def forward(self, images):
        """
        images: Tensor [B, 3, H, W] in [-1, 1]
        Returns: Tensor [B, 2048] feature vectors
        """
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        images = (images + 1) / 2  # [-1, 1] → [0, 1]
        images = self.resize_op(images)
        return self.model(images.to(self.device))
    

from sklearn.neighbors import NearestNeighbors
import numpy as np

def compute_precision_recall(real_feats, fake_feats, k=3, radius=None):
    """
    real_feats: [N_real, D]
    fake_feats: [N_fake, D]
    """
    real_feats = real_feats.cpu().numpy()
    fake_feats = fake_feats.cpu().numpy()

    # Fit nearest neighbor model on real
    nbrs_real = NearestNeighbors(n_neighbors=k).fit(real_feats)
    distances_real, _ = nbrs_real.kneighbors(real_feats)
    real_radii = distances_real[:, -1]  # radius = distance to kth neighbor

    # Recall: how many real samples are "covered" by fakes
    nbrs_fake = NearestNeighbors(n_neighbors=1).fit(fake_feats)
    distances_to_fake, _ = nbrs_fake.kneighbors(real_feats)
    recall = np.mean(distances_to_fake[:, 0] <= real_radii)

    # Precision: how many fake samples lie within real manifold
    distances_to_real, _ = nbrs_real.kneighbors(fake_feats)
    precision = np.mean(distances_to_real[:, -1] <= real_radii.mean())

    return precision, recall

