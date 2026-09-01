"""Secondary model (PyTorch): an LSTM over the raw [seq_len, 4] sensor window.

Channels: temp_c, humidity_pct, tilt_deg, distance_mm. The network sees the raw
daily sequence a field node would stream, standardised with train-set stats.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .. import RISK_CLASSES
from ..config import Config
from ..utils import get_logger, pick_torch_device

log = get_logger("minesub.models.torch")


class _LSTMNet(nn.Module):
    def __init__(self, n_ch: int, hidden: int, layers: int, dropout: float, n_cls: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_ch, hidden_size=hidden, num_layers=layers,
            batch_first=True, dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_cls),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])          # last timestep -> logits


class LSTMRiskModel:
    def __init__(self, cfg: Config, task: str = "classification"):
        self.cfg = cfg
        self.p = cfg["model"]["torch"]
        self.task = task
        self.n_out = 1 if task == "regression" else len(RISK_CLASSES)
        self.device = pick_torch_device()
        self.net: _LSTMNet | None = None
        self.mu: np.ndarray | None = None
        self.sd: np.ndarray | None = None
        self.y_mu: float = 0.0     # regression target standardisation
        self.y_sd: float = 1.0

    # -- helpers ----------------------------------------------------------
    def _standardise(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mu) / self.sd

    def _loader(self, X, y, shuffle):
        tx = torch.tensor(self._standardise(X), dtype=torch.float32)
        if self.task == "regression":
            ty = torch.tensor((np.asarray(y, float) - self.y_mu) / self.y_sd,
                              dtype=torch.float32).unsqueeze(1)
        else:
            ty = torch.tensor(y, dtype=torch.long)
        return DataLoader(TensorDataset(tx, ty), batch_size=int(self.p["batch_size"]),
                          shuffle=shuffle)

    # -- train ----------------------------------------------------------
    def fit(self, Xtr, ytr, Xva, yva):
        torch.manual_seed(int(self.cfg["seed"]))
        self.mu = Xtr.reshape(-1, Xtr.shape[-1]).mean(0)
        self.sd = Xtr.reshape(-1, Xtr.shape[-1]).std(0) + 1e-6
        reg = self.task == "regression"
        if reg:
            self.y_mu = float(np.mean(ytr))
            self.y_sd = float(np.std(ytr) + 1e-9)

        self.net = _LSTMNet(
            n_ch=Xtr.shape[-1], hidden=int(self.p["hidden_size"]),
            layers=int(self.p["num_layers"]), dropout=float(self.p["dropout"]),
            n_cls=self.n_out,
        ).to(self.device)

        if reg:
            crit = nn.SmoothL1Loss()          # Huber: robust to the heavy tail
        else:
            cls_count = np.bincount(ytr, minlength=len(RISK_CLASSES)).astype(float)
            cls_w = torch.tensor(cls_count.sum() / (len(cls_count) * np.maximum(cls_count, 1)),
                                 dtype=torch.float32, device=self.device)
            crit = nn.CrossEntropyLoss(weight=cls_w)
        opt = torch.optim.Adam(self.net.parameters(), lr=float(self.p["lr"]),
                               weight_decay=float(self.p["weight_decay"]))

        tr_loader = self._loader(Xtr, ytr, shuffle=True)
        best_score, best_state, bad = -np.inf, None, 0
        for epoch in range(1, int(self.p["max_epochs"]) + 1):
            self.net.train()
            for xb, yb in tr_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                loss = crit(self.net(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
                opt.step()

            raw = self._predict_raw(Xva)
            if reg:
                pred = raw.ravel() * self.y_sd + self.y_mu
                mae = float(np.mean(np.abs(pred - np.asarray(yva, float))))
                score, tag, shown = -mae, "val_MAE", mae
            else:
                pred = raw.argmax(1)
                f1 = f1_score(yva, pred, average="macro")
                score, tag, shown = f1, "val_macroF1", f1
            if score > best_score + 1e-4:
                best_score, best_state, bad = score, {k: v.cpu().clone()
                                                      for k, v in self.net.state_dict().items()}, 0
            else:
                bad += 1
            if epoch % 5 == 0 or bad == 0:
                log.info("epoch %3d  %s=%.4f  (best=%.4f)", epoch, tag, shown, abs(best_score))
            if bad >= int(self.p["patience"]):
                log.info("early stop at epoch %d", epoch)
                break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        log.info("done; best %s=%.4f  device=%s",
                 "val_MAE" if reg else "val macro-F1", abs(best_score), self.device)
        return self

    # -- inference ----------------------------------------------------------
    @torch.no_grad()
    def _predict_raw(self, X) -> np.ndarray:
        self.net.eval()
        tx = torch.tensor(self._standardise(X), dtype=torch.float32, device=self.device)
        out = []
        for i in range(0, len(tx), 1024):
            out.append(self.net(tx[i:i + 1024]).cpu().numpy())
        return np.concatenate(out, 0)

    _predict_logits = _predict_raw       # backwards-compatible alias

    def predict_proba(self, X) -> np.ndarray:
        logits = torch.tensor(self._predict_raw(X))
        return torch.softmax(logits, dim=1).numpy()

    def predict(self, X) -> np.ndarray:
        """Regression: de-standardised scalar prediction per sample."""
        return self._predict_raw(X).ravel() * self.y_sd + self.y_mu

    # -- persistence --------------------------------------------------------
    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.net.state_dict(),
            "mu": self.mu, "sd": self.sd,
            "task": self.task, "n_out": self.n_out,
            "y_mu": self.y_mu, "y_sd": self.y_sd,
            "arch": {"hidden": int(self.p["hidden_size"]),
                     "layers": int(self.p["num_layers"]),
                     "dropout": float(self.p["dropout"]),
                     "n_ch": len(self.mu)},
            "classes": list(RISK_CLASSES),
        }, path)
        log.info("saved -> %s", path)

    @classmethod
    def load(cls, cfg: Config, path: str | Path) -> "LSTMRiskModel":
        # our own artifact (contains numpy mu/sd arrays) -> full unpickle
        blob = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls(cfg, task=blob.get("task", "classification"))
        a = blob["arch"]
        obj.n_out = int(blob.get("n_out", len(RISK_CLASSES)))
        obj.net = _LSTMNet(a["n_ch"], a["hidden"], a["layers"], a["dropout"],
                           obj.n_out).to(obj.device)
        obj.net.load_state_dict(blob["state_dict"])
        obj.mu, obj.sd = blob["mu"], blob["sd"]
        obj.y_mu, obj.y_sd = float(blob.get("y_mu", 0.0)), float(blob.get("y_sd", 1.0))
        return obj
