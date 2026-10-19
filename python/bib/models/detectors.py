"""One-class anomaly detectors. All share fit(X_genuine) / score(X) with
higher score = more anomalous (more likely an impostor).

Distance baselines follow Killourhy & Maxion (DSN 2009) exactly, so the
benchmark harness can be validated against their published EERs before any
new model is trusted:
  Euclidean            squared distance to the training mean
  ManhattanScaled      sum_i |x_i - mean_i| / mad_i  (mad = mean absolute deviation)
  MahalanobisNN        min Mahalanobis distance to any training vector
Learned:
  DenseAE              MLP autoencoder (PyTorch); score = reconstruction MSE
                       in standardized space; optional seed ensemble
  SeqAE (seq_ae.py)    key-conditioned LSTM autoencoder for keystroke sequences
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


class _Imputer:
    def fit(self, X):
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN column: group never seen in enrollment
            self.mu = np.nanmean(X, axis=0)
        self.mu = np.where(np.isnan(self.mu), 0.0, self.mu)
        return self

    def __call__(self, X):
        X = np.array(X, dtype=np.float64, copy=True)
        r, c = np.nonzero(np.isnan(X))
        X[r, c] = self.mu[c]
        return X


class Euclidean:
    name = "Euclidean"

    def fit(self, X):
        self.imp = _Imputer().fit(X)
        self.mu = self.imp(X).mean(0)
        return self

    def score(self, X):
        return ((self.imp(X) - self.mu) ** 2).sum(1)


class ManhattanScaled:
    name = "Manhattan (scaled)"

    def fit(self, X):
        self.imp = _Imputer().fit(X)
        X = self.imp(X)
        self.mu = X.mean(0)
        self.mad = np.abs(X - self.mu).mean(0)
        self.mad[self.mad == 0] = 1e-9
        return self

    def score(self, X):
        return (np.abs(self.imp(X) - self.mu) / self.mad).sum(1)


class MahalanobisNN:
    name = "Nearest neighbor (Mahalanobis)"

    def fit(self, X):
        self.imp = _Imputer().fit(X)
        self.X = self.imp(X)
        self.P = np.linalg.pinv(np.cov(self.X, rowvar=False))
        return self

    def score(self, X):
        X = self.imp(X)
        D = X[:, None, :] - self.X[None, :, :]
        d2 = np.einsum("nmi,ij,nmj->nm", D, self.P, D)
        return np.sqrt(np.maximum(d2.min(1), 0))


@dataclass
class AEConfig:
    hidden: int = 32
    bottleneck: int = 8
    epochs: int = 400
    lr: float = 3e-3
    weight_decay: float = 1e-4
    noise: float = 0.1          # denoising: Gaussian noise (std, standardized units) on inputs
    n_seeds: int = 3            # ensemble members (scores averaged)
    transform: str = "log"      # "log" (log of positive timings) or "none"
    robust_scale: bool = True   # median/IQR instead of mean/std
    score_norm: str = "l2"      # "l2": mean squared residual; "l1": mean absolute residual (robust, like Manhattan)


class _MLP(nn.Module):
    def __init__(self, d, h, b):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, h), nn.Tanh(), nn.Linear(h, b))
        self.dec = nn.Sequential(nn.Tanh(), nn.Linear(b, h), nn.Tanh(), nn.Linear(h, d))

    def forward(self, x):
        return self.dec(self.enc(x))


class DenseAE:
    name = "Autoencoder (ours)"

    def __init__(self, cfg: AEConfig | None = None, seed: int = 0):
        self.cfg = cfg or AEConfig()
        self.seed = seed

    def _tf(self, X):
        X = self.imp(X)
        if self.cfg.transform == "log":
            X = np.sign(X) * np.log1p(np.abs(X) * 1000.0)  # timings in s: log1p(ms), sign-safe for UD < 0
        return (X - self.loc) / self.scale

    def fit(self, X):
        cfg = self.cfg
        self.imp = _Imputer().fit(X)
        Xi = self.imp(X)
        if cfg.transform == "log":
            Xi = np.sign(Xi) * np.log1p(np.abs(Xi) * 1000.0)
        if cfg.robust_scale:
            self.loc = np.median(Xi, 0)
            q = np.quantile(Xi, [0.25, 0.75], axis=0)
            self.scale = (q[1] - q[0]) / 1.349
        else:
            self.loc, self.scale = Xi.mean(0), Xi.std(0)
        self.scale = np.where(self.scale > 1e-9, self.scale, 1.0)
        Z = torch.tensor((Xi - self.loc) / self.scale, dtype=torch.float32)
        self.nets = []
        for k in range(cfg.n_seeds):
            g = torch.Generator().manual_seed(self.seed * 1000 + k)
            torch.manual_seed(self.seed * 1000 + k)
            net = _MLP(Z.shape[1], cfg.hidden, cfg.bottleneck)
            opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
            for _ in range(cfg.epochs):  # full batch: enrollment sets are a few hundred rows
                inp = Z + cfg.noise * torch.randn(Z.shape, generator=g) if cfg.noise > 0 else Z
                loss = ((net(inp) - Z) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            net.eval()
            self.nets.append(net)
        return self

    @torch.no_grad()
    def score(self, X):
        Z = torch.tensor(self._tf(X), dtype=torch.float32)
        s = np.zeros(len(Z))
        for net in self.nets:
            r = net(Z) - Z
            s += (r.abs() if self.cfg.score_norm == "l1" else r ** 2).mean(1).numpy()
        return s / len(self.nets)

    def state(self):
        return {"cfg": self.cfg.__dict__, "imp_mu": self.imp.mu, "loc": self.loc, "scale": self.scale,
                "nets": [n.state_dict() for n in self.nets]}

    @classmethod
    def from_state(cls, st):
        m = cls(AEConfig(**st["cfg"]))
        m.imp = _Imputer()
        m.imp.mu, m.loc, m.scale = st["imp_mu"], st["loc"], st["scale"]
        m.nets = []
        for sd in st["nets"]:
            d = len(m.loc)
            n = _MLP(d, m.cfg.hidden, m.cfg.bottleneck)
            n.load_state_dict(sd)
            n.eval()
            m.nets.append(n)
        return m


DETECTORS = {"euclidean": Euclidean, "manhattan_scaled": ManhattanScaled, "mahalanobis_nn": MahalanobisNN,
             "ae": DenseAE}
