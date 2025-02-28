import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
from collections import OrderedDict
from typing import Tuple
import numpy as np


class TokenSuppression(nn.Module):

    def __init__(self, d_model: int, K: int = 8):
        super().__init__()
        self.d_model = d_model
        self.K = K

        self.register_buffer('pc_components', torch.randn(K, d_model))
        self.pc_components.data = F.normalize(self.pc_components, dim=1)

        self.beta = 3.0
        self.gamma = 5.0
        self.alpha = 0.9

    def update_pca(self, foreground_tokens: torch.Tensor):
        with torch.no_grad():
            cov = torch.matmul(foreground_tokens.T, foreground_tokens) / foreground_tokens.size(0)
            _, eigenvectors = torch.linalg.eigh(cov)
            self.pc_components.copy_(eigenvectors[:, -self.K:].T)

    def forward(self, x: torch.Tensor, attn_weights: torch.Tensor, text_cls: torch.Tensor):

        seq_len, batch_size, _ = x.shape

        # 分离CLS Token与图像Patch Tokens
        cls_token = x[0]
        patch_tokens = x[1:]


        cls_attn = attn_weights[:, :, 0, 1:].mean(dim=1)


        mu = cls_attn.mean(dim=1, keepdim=True)
        sigma = cls_attn.std(dim=1, keepdim=True)
        fg_mask = (cls_attn > mu + 1.5 * sigma).float().permute(1, 0)


        text_sim = F.cosine_similarity(
            patch_tokens,
            text_cls.unsqueeze(0),
            dim=-1
        )
        semantic_mismatch = 1.0 - torch.relu(text_sim)

        token_norm = torch.norm(patch_tokens, dim=-1)  # [seq_len-1, batch]

        anomaly_score = token_norm * semantic_mismatch  # [seq_len-1, batch]


        bg_mask = (fg_mask < 0.5)
        bg_scores = anomaly_score[bg_mask]

        if bg_scores.numel() > 0:
            mu_bg = bg_scores.mean()
            sigma_bg = bg_scores.std()
            threshold = mu_bg + self.beta * sigma_bg
        else:
            threshold = anomaly_score.max()  # 无背景时跳过抑制

        # Sigmoid门控
        gate = torch.sigmoid(self.gamma * (threshold - anomaly_score))
        gate = fg_mask + (1 - fg_mask) * gate  # 前景区域保持gate=1

        proj_coeff = torch.matmul(patch_tokens, self.pc_components.T)
        proj_tokens = torch.matmul(proj_coeff, self.pc_components)


        gate = gate.unsqueeze(-1)
        suppressed_tokens = gate * patch_tokens + (1 - gate) * proj_tokens

        suppressed_x = torch.cat([cls_token.unsqueeze(0), suppressed_tokens], dim=0)

        return suppressed_x





