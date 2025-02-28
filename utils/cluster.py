import torch
import torch.nn.functional as F
from torch import nn

def dynamic_clustering(self, similarities):
    B, M = similarities.shape

    dists = torch.cdist(
        similarities.unsqueeze(-1),  # (B,M,1)
        self.cluster_centers.unsqueeze(0).unsqueeze(0),  # (1,1,K,D)
        p=2
    )  # (B,M,K)

    cluster_probs = F.softmax(-dists, dim=-1)

    weighted_sims = similarities.unsqueeze(-1) * cluster_probs  # (B,M,K)
    cluster_means = weighted_sims.sum(dim=1) / (cluster_probs.sum(dim=1) + 1e-6)  # (B,K)
    cluster_stds = torch.sqrt(
        ((weighted_sims - cluster_means.unsqueeze(1)) ** 2 * cluster_probs).sum(dim=1)
        / (cluster_probs.sum(dim=1) + 1e-6)
    )  # (B,K)

    return cluster_means, cluster_stds  # (B,K), (B,K)