"""
Zero-Inflated Negative Binomial (ZINB) Loss

Implements ZINB loss function for modeling single-cell RNA-seq data
with proper handling of sparsity and overdispersion.
"""

import torch
import torch.nn as nn


class ZINBLoss(nn.Module):
    """
    Zero-Inflated Negative Binomial Loss for single-cell RNA-seq data
    Handles sparsity in single-cell data
    """
    def __init__(self):
        super(ZINBLoss, self).__init__()

    def forward(self, mu, theta, pi, target):
        """
        Args:
            mu: mean of negative binomial (batch_size, n_genes)
            theta: dispersion parameter (batch_size, n_genes)
            pi: zero-inflation probability (batch_size, n_genes)
            target: true expression values (batch_size, n_genes)
        """
        eps = 1e-10

        # Negative binomial component
        theta = torch.clamp(theta, min=eps)
        t1 = torch.lgamma(theta + eps) + torch.lgamma(target + 1.0) - torch.lgamma(target + theta + eps)
        t2 = (theta + target) * torch.log(1.0 + (mu / (theta + eps))) + (target * (torch.log(theta + eps) - torch.log(mu + eps)))
        nb_case = t1 + t2 - torch.log(1.0 - pi + eps)

        # Zero-inflation component
        zero_nb = torch.pow(theta / (theta + mu + eps), theta)
        zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + eps)

        # Combine cases
        result = torch.where(target < eps, zero_case, nb_case)

        return result.mean()
