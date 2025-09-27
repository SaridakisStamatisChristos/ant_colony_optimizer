from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


class BaseNeuralNetwork(nn.Module):
    """Adds device/dtype plumbing + simple (de)serialization helpers."""

    def __init__(
        self,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self._dtype = dtype
        self._device = device or torch.device("cpu")

    def to_device(self) -> "BaseNeuralNetwork":
        return self.to(dtype=self._dtype, device=self._device)

    # Helpers that always reflect the real module state (after any .to(...))
    @property
    def param_device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return self._device

    @property
    def param_dtype(self) -> torch.dtype:
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            return self._dtype

    def save_model(self, path: str) -> None:
        torch.save(
            {
                "state_dict": self.state_dict(),
                "dtype": self._dtype,
                "device": str(self._device),
            },
            path,
        )

    def load_model(self, path: str) -> None:
        blob = torch.load(path, map_location="cpu")
        self.load_state_dict(blob["state_dict"])
        self._dtype = blob.get("dtype", torch.float32)
        self._device = torch.device(blob.get("device", "cpu"))
        self.to_device().eval()


class RiskAssessmentNetwork(BaseNeuralNetwork):
    def __init__(
        self,
        n_assets: int,
        hidden: Optional[List[int]] = None,
        **kw,
    ) -> None:
        super().__init__(**kw)
        hidden = hidden or [96, 64, 32]  # avoid mutable default
        dims = [n_assets] + hidden + [n_assets]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x.to(device=self.param_device, dtype=self.param_dtype))


class PheromoneNetwork(BaseNeuralNetwork):
    def __init__(
        self,
        n_assets: int,
        embed_dim: int = 48,
        heads: int = 4,
        **kw,
    ) -> None:
        super().__init__(**kw)
        self.n_assets = n_assets
        self.emb = nn.Embedding(n_assets, embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads=heads, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 96),
            nn.ReLU(),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Linear(48, n_assets),
        )

    def transition_matrix(self) -> torch.Tensor:
        idx = torch.arange(self.n_assets, device=self.param_device)
        x = self.emb(idx).unsqueeze(0)
        a, _ = self.attn(x, x, x)
        h = self.norm(a + x)
        logits = self.ffn(h)
        return torch.softmax(logits, dim=-1).squeeze(0).to(dtype=self.param_dtype)

    def forward(self, *_args, **_kw) -> torch.Tensor:  # back-compat
        return self.transition_matrix()

