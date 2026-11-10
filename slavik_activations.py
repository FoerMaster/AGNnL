"""Shared SLA'vik activation modules."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class SlavikA(torch.nn.Module):
    """Trainable SLA'vik-A activation with parameters along a chosen axis."""

    def __init__(self, width: int, channel_dim: int = -1):
        super().__init__()
        self.width = width
        self.channel_dim = channel_dim
        self.kappa_raw = torch.nn.Parameter(torch.full((width,), 0.5413))
        self.lambda_raw = torch.nn.Parameter(torch.full((width,), -1.7346))
        self.alpha = torch.nn.Parameter(torch.full((width,), 0.37))
        self.beta = torch.nn.Parameter(torch.full((width,), 0.11))

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        dim = self.channel_dim % a.ndim
        if a.shape[dim] != self.width:
            raise ValueError(
                f"Expected width {self.width} on dimension {dim}, got {a.shape[dim]}"
            )
        shape = [1] * a.ndim
        shape[dim] = self.width
        kappa = F.softplus(self.kappa_raw).view(shape)
        lam = torch.sigmoid(self.lambda_raw).view(shape)
        alpha = self.alpha.view(shape)
        beta = self.beta.view(shape)
        b = alpha * a + beta
        s = a + b
        p = a * b
        return (torch.clamp(s, min=0)
                + lam * torch.clamp(s, max=0) / (1 + s.abs())
                + kappa * p / (1 + p.abs()))


class SlavikR2(torch.nn.Module):
    """Trainable ReLU-based SLA'vik-R2 activation."""

    def __init__(self, width: int, g0: float = 0.5):
        super().__init__()
        self.width = width
        self.channel_dim = -1
        self.kappa_raw = torch.nn.Parameter(torch.full((width,), 0.5413))
        self.lambda_raw = torch.nn.Parameter(torch.full((width,), -1.7346))
        self.alpha = torch.nn.Parameter(torch.full((width,), 0.37))
        self.beta = torch.nn.Parameter(torch.full((width,), 0.11))
        raw_g = math.log(g0 / (1 - g0))
        self.g_raw = torch.nn.Parameter(torch.full((width,), raw_g))

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        if a.shape[-1] != self.width:
            raise ValueError(
                f"Expected width {self.width} on final dimension, got {a.shape[-1]}"
            )
        kappa = F.softplus(self.kappa_raw)
        lam = torch.sigmoid(self.lambda_raw)
        g = torch.sigmoid(self.g_raw)
        b = self.alpha * a + self.beta
        p = a * b
        correction = (kappa * p / (1 + p.abs())
                      + lam * torch.clamp(a, max=0) / (1 + a.abs()))
        return torch.relu(a) + g * correction
