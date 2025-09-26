from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from .utils import safe_softmax

class BaseNeuralNetwork(nn.Module):
    def save_model(self, filepath: str) -> None:
        torch.save(self.state_dict(), filepath)

    def load_model(self, filepath: str) -> None:
        self.load_state_dict(torch.load(filepath, map_location="cpu"))
        self.eval()

class RiskAssessmentNetwork(BaseNeuralNetwork):
    def __init__(self, n_assets: int, hidden: list[int] = [96,64,32]):
        super().__init__()
        self.n_assets = n_assets
        in_dim = n_assets * 3
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden[0]), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden[1], hidden[2]), nn.ReLU(),
            nn.Linear(hidden[2], n_assets), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @torch.no_grad()
    def predict(self, mu: np.ndarray, sigma: np.ndarray, corr: np.ndarray) -> np.ndarray:
        avg_corr = (corr.sum(axis=1) - 1.0) / max(self.n_assets - 1, 1)
        feats = np.stack([mu, sigma, avg_corr], axis=1).reshape(-1).astype(np.float32)
        tens = torch.from_numpy(feats).unsqueeze(0)
        return self.forward(tens).cpu().numpy().ravel()

class PheromoneNetwork(BaseNeuralNetwork):
    def __init__(self, n_assets: int, embedding_dim: int = 48):
        super().__init__()
        self.n_assets = n_assets
        self.asset_embedding = nn.Embedding(n_assets, embedding_dim)
        self.attn = nn.MultiheadAttention(embedding_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, 96), nn.ReLU(),
            nn.Linear(96, 48), nn.ReLU(),
            nn.Linear(48, n_assets * n_assets), nn.Sigmoid()
        )

    def forward(self, asset_indices: torch.Tensor) -> torch.Tensor:
        emb = self.asset_embedding(asset_indices).unsqueeze(0)
        attn_out, _ = self.attn(emb, emb, emb)
        emb = self.norm(emb + attn_out)
        pooled = emb.mean(dim=1)
        mat = self.ffn(pooled).view(self.n_assets, self.n_assets)
        return mat

    @torch.no_grad()
    def transition_probs(self, current: int, visited: list[int]) -> np.ndarray:
        idx = torch.arange(self.n_assets, dtype=torch.long)
        mat = self.forward(idx).cpu().numpy()
        row = mat[current].copy()
        mask = np.ones(self.n_assets, dtype=bool)
        mask[current] = False
        if visited:
            mask[visited] = False
        row = row * mask.astype(float)
        if row.sum() <= 0:
            p = np.ones(self.n_assets, dtype=float) * mask.astype(float)
            s = p.sum()
            return p / s if s > 0 else np.ones(self.n_assets) / self.n_assets
        return safe_softmax(row)
