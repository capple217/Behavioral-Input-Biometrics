"""Key-conditioned LSTM autoencoder for keystroke timing sequences.

Input per keystroke: token (which key) + [H, DD, UD, boundary] (squashed ms).
  encoder: embedding(key) ++ timings -> BiLSTM -> final states -> z (bottleneck)
  decoder: embedding(key) ++ z (repeated) -> LSTM -> predicted [H, DD, UD]
Loss / anomaly score: MSE on the timing channels, with DD/UD masked at
segment boundaries (first key, after pauses), averaged per window.

Why condition the decoder on the keys: in free text the key sequence is
arbitrary, and identity of a typist lives in *how long* each key / transition
takes given the keys. Feeding the keys to the decoder means the bottleneck only
has to carry timing style, and reconstruction error measures "these timings
are unusual for this person on these keys".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from bib.features.keystroke import VOCAB


@dataclass
class SeqAEConfig:
    emb: int = 8
    hidden: int = 32
    bottleneck: int = 8
    epochs: int = 120
    lr: float = 3e-3
    batch: int = 64
    noise: float = 0.05
    weight_decay: float = 1e-5
    n_seeds: int = 1


class _Net(nn.Module):
    def __init__(self, cfg: SeqAEConfig):
        super().__init__()
        self.emb = nn.Embedding(len(VOCAB), cfg.emb)
        self.enc = nn.LSTM(cfg.emb + 4, cfg.hidden, batch_first=True, bidirectional=True)
        self.to_z = nn.Linear(2 * cfg.hidden, cfg.bottleneck)
        self.dec = nn.LSTM(cfg.emb + cfg.bottleneck, cfg.hidden, batch_first=True)
        self.out = nn.Linear(cfg.hidden, 3)

    def forward(self, tok, x):
        e = self.emb(tok)
        _, (h, _) = self.enc(torch.cat([e, x], dim=2))
        z = torch.tanh(self.to_z(torch.cat([h[0], h[1]], dim=1)))
        y, _ = self.dec(torch.cat([e, z.unsqueeze(1).expand(-1, tok.shape[1], -1)], dim=2))
        return self.out(y)


class SeqAE:
    name = "LSTM autoencoder (ours)"

    def __init__(self, cfg: SeqAEConfig | None = None, seed: int = 0):
        self.cfg = cfg or SeqAEConfig()
        self.seed = seed

    def _prep(self, x):
        x = np.array(x, dtype=np.float32, copy=True)
        mask = np.ones(x.shape[:2] + (3,), dtype=np.float32)
        mask[:, :, 1:] = 1.0 - x[:, :, 3:4]           # DD/UD invalid at boundaries
        x[:, :, :3] = (x[:, :, :3] - self.mu) / self.sd
        x[:, :, 1:3] *= mask[:, :, 1:]                  # masked entries -> 0 after scaling
        return torch.tensor(x), torch.tensor(mask)

    def fit(self, tok, x):
        cfg = self.cfg
        valid = np.concatenate([x[:, :, 0:1], x[:, :, 1:3] * np.where(x[:, :, 3:4] > 0, np.nan, 1.0)], axis=2)
        self.mu = np.nanmean(valid.reshape(-1, 3), 0).astype(np.float32)
        self.sd = np.nanstd(valid.reshape(-1, 3), 0).astype(np.float32) + 1e-6
        T = torch.tensor(np.asarray(tok), dtype=torch.long)
        X, M = self._prep(x)
        self.nets = []
        for k in range(cfg.n_seeds):
            torch.manual_seed(self.seed * 1000 + k)
            g = torch.Generator().manual_seed(self.seed * 1000 + k)
            net = _Net(cfg)
            opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
            n = len(T)
            for _ in range(cfg.epochs):
                perm = torch.randperm(n, generator=g)
                for i in range(0, n, cfg.batch):
                    b = perm[i:i + cfg.batch]
                    xb = X[b].clone()
                    if cfg.noise > 0:
                        xb[:, :, :3] += cfg.noise * torch.randn(xb[:, :, :3].shape, generator=g) * M[b]
                    pred = net(T[b], xb)
                    loss = (((pred - X[b, :, :3]) ** 2) * M[b]).sum() / M[b].sum()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            net.eval()
            self.nets.append(net)
        return self

    @torch.no_grad()
    def score(self, tok, x):
        T = torch.tensor(np.asarray(tok), dtype=torch.long)
        X, M = self._prep(x)
        s = np.zeros(len(T))
        for net in self.nets:
            err = ((net(T, X) - X[:, :, :3]) ** 2) * M
            s += (err.sum((1, 2)) / M.sum((1, 2))).numpy()
        return s / len(self.nets)

    def state(self):
        return {"cfg": self.cfg.__dict__, "mu": self.mu, "sd": self.sd,
                "nets": [n.state_dict() for n in self.nets]}

    @classmethod
    def from_state(cls, st):
        m = cls(SeqAEConfig(**st["cfg"]))
        m.mu, m.sd = st["mu"], st["sd"]
        m.nets = []
        for sd in st["nets"]:
            n = _Net(m.cfg)
            n.load_state_dict(sd)
            n.eval()
            m.nets.append(n)
        return m
